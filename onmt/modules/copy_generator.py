import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute

import numpy as np


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map, siamese_attn=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
           siamese_attn(FloatTensor): use with copy_prob.
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.src_indicator = None

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "siamese_attn_0": attns.get("siamese")[0],
            "siamese_attn_1": attns.get("siamese")[1],
            "align": batch.alignment[range_[0] + 1: range_[1]]
        })
        return shard_state

    def _compute_siamese_loss(self, batch, siamese_attn_0, siamese_attn_1):

        max_input_seq_len = 400

        # if self.src_indicator is None:
        #     use_cuda = torch.cuda.is_available()
        #     self.src_indicator = torch.cuda.LongTensor(batch.batch_size, max_input_seq_len, self.criterion.vocab_size) if use_cuda else torch.LongTensor

        def _set_src_indicator(src):
            for sent_id in range(src.size(1)):
                self.src_indicator[sent_id].fill_(0)
                for word_id in range(src.size(0)):
                    self.src_indicator[sent_id, word_id, src[word_id, sent_id, 0]] = 1

        def _get_src_indicator(src):
            src_indicator = np.zeros((batch.batch_size, max_input_seq_len, self.criterion.vocab_size))
            for sent_id in range(src.size(1)):
                for word_id in range(src.size(0)):
                    src_indicator[sent_id, word_id, src[word_id, sent_id, 0]] = 1
            return torch.from_numpy(src_indicator)

        src, _ = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)

        # _set_src_indicator(src)
        src_indicator = _get_src_indicator(src).to(siamese_attn_1.device)

        padded_siamese_attn_0 = torch.zeros(
            size=(siamese_attn_0.size()[0], max_input_seq_len, siamese_attn_0.size()[2]), device=siamese_attn_0.device,
            dtype=siamese_attn_0.dtype, requires_grad=True).clone()
        padded_siamese_attn_0[:, :siamese_attn_0.size()[1], :] = siamese_attn_0
        v1 = torch.diagonal(input=torch.bmm(padded_siamese_attn_0, src_indicator.transpose(1, 2).float()), dim1=-1, dim2=-2).float()

        padded_siamese_attn_1 = torch.zeros(
            size=(siamese_attn_1.size()[0], max_input_seq_len, siamese_attn_1.size()[2]), device=siamese_attn_1.device,
            dtype=siamese_attn_1.dtype, requires_grad=True).clone()
        padded_siamese_attn_1[:, :siamese_attn_1.size()[1], :] = siamese_attn_1
        v2 = torch.diagonal(input=torch.bmm(padded_siamese_attn_1, src_indicator.transpose(1, 2).float()), dim1=-1, dim2=-2)

        vocab = batch.dataset.fields['src'].base_field.vocab
        src_ex_vocab = batch.src_ex_vocab
        v0 = torch.zeros(v1.size())
        for batch_id in range(src.size()[1]):
            for i, word_id in enumerate(src[:, batch_id, 0]):
                word_str = vocab.itos[word_id]
                if vocab.freqs[word_str] == 0:
                    v0[batch_id, i] = 0
                else:
                    v0[batch_id, i] = src_ex_vocab[batch_id].freqs[word_str] / float(vocab.freqs[word_str])
        v0 = v0.to(siamese_attn_1.device)

        return torch.abs(v0 - v1 + v2).sum()

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None, siamese_attn_0=None, siamese_attn_1=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        if siamese_attn_0 is not None and siamese_attn_1 is not None:
            siamese_loss = self._compute_siamese_loss(batch, siamese_attn_0, siamese_attn_1)
            print(loss.size(), siamese_loss.size())
            loss += siamese_loss

        return loss, stats
