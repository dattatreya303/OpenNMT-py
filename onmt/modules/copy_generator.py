""" Generator module """
import torch.nn as nn
import torch
import torch.cuda
import numpy as np

import torch.nn.functional as F

import onmt.inputters as inputters
from onmt.utils.misc import aeq
from onmt.utils import loss


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

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
       tgt_dict (Vocab): output target dictionary

    """

    def __init__(self,
                 input_size,
                 tgt_dict,
                 gumbel_tags=False,
                 start_annealing_steps=1000,
                 start_normalizing_temp=1.0,
                 min_normalizing_temp=0.2,
                 annealing_factor=1e-3):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # Content Selector options
        self.gumbel_tags = gumbel_tags
        self.normalizing_temp = start_normalizing_temp
        self.start_normalizing_temp = start_normalizing_temp
        self.min_normalizing_temp = min_normalizing_temp
        self.start_annealing_steps = start_annealing_steps
        self.annealing_factor = annealing_factor

        # Initialize a counter for annealing the temperature
        self.annealing_steps = 0

        # Log all the things
        if self.gumbel_tags:
            print("###########################################")
            print("USING GUMBEL SAMPLING FOR COPY ATTENTION")
            print("ACTIVATE AFTER {} STEPS".format(self.start_annealing_steps))
            print("START TEMP:", self.start_normalizing_temp)
            print("ANNEALING FACTOR:", self.annealing_factor)
            print("MIN TEMP:", self.min_normalizing_temp)
            print("###########################################")

    def forward(self, hidden, attn, tags, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[inputters.PAD_WORD]] = -float('inf')
        prob = self.softmax(logits)
        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)


        # GUMBEL SOFTMAX PREDICTED MASK
        # Todo: try masking attention in general, not only here
        if self.gumbel_tags and (self.annealing_steps >= self.start_annealing_steps):
            # tag_out_pre is slen x batch_size x 2
            # attn is batch*tlen x slen
            # Therefore we switch dimensions and expand first dimension
            tag_out_pre = self._gumbel_sample(tags)
            # Target length
            tlen = int(batch_by_tlen/batch)
            tag_out = tag_out_pre.transpose(0, 1)\
                                 .unsqueeze(0)\
                                 .expand(tlen, batch, slen)\
                                 .contiguous()\
                                 .view(-1, slen)
            # finally mask the attention
            mul_attn = torch.mul(tag_out, attn)
            # renormalize (todo: with temperature)
            mul_attn = F.softmax(mul_attn/1., -1)
        else:
            # set variables for non-gumbel version
            tag_out_pre = tags
            mul_attn = attn

        self.annealing_steps += 1

        # Normal p_copy from here
        mul_attn = torch.mul(mul_attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        extra_stats = {'gumbel_temp': [self.normalizing_temp],
                       'avg_copy_prob': [p_copy.mean()],
                       'avg_content_selection': [tag_out_pre.mean()]}
        return torch.cat([out_prob, copy_prob], 1), tag_out_pre, extra_stats

    def _gumbel_sample(self, tags):
        src_len, bsize, tsize = tags.shape
        # Flatten batched output
        flat_tags = tags.view(-1, tsize)
        # Sample noise
        U = torch.rand(flat_tags.shape)
        eps = 1e-20
        U = -torch.log(-torch.log(U + eps) + eps).to(tags.device)
        # Apply temperature
        x = (flat_tags + U) / self.normalizing_temp
        x = F.softmax(x, dim=-1)
        if self.normalizing_temp > self.min_normalizing_temp * 1.01:
            self._anneal_temperature()
        return x.view_as(tags)[:,:,1]

    def _anneal_temperature(self):
        if self.annealing_steps % 5 == 0:
            self.normalizing_temp = max(self.min_normalizing_temp,
                                        self.start_normalizing_temp * np.exp(
                -self.annealing_factor * (self.annealing_steps - self.start_annealing_steps)))
            print("annealing temperature to {:2f}".format(self.normalizing_temp))


class CopyGeneratorCriterion(object):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss


class CopyTagCriterion(object):
    def __init__(self, pad, eps=1e-10):
        self.eps = eps
        self.pad = pad
        self.crit = torch.nn.CrossEntropyLoss()

    def __call__(self, yhat, y, src_lengths):
        # print(yhat.shape, y.shape)
        loss = self.crit(yhat, y)
        for s, t in zip(y[:15], F.softmax(yhat[:15])):
            print("{} {:.2f}".format(s.item(), t[1].item()))
        return loss


class CopyGeneratorLossCompute(loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length,
                 eps=1e-20,
                 supervise_tags=False,
                 gumbel_tags=True,
                 normalizing_temp=0.1):
        super(CopyGeneratorLossCompute, self).__init__(
            generator, tgt_vocab)
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(
            len(tgt_vocab),
            force_copy,
            self.padding_idx)
        self.tag_criterion = CopyTagCriterion(self.padding_idx)
        self.supervise_tags = supervise_tags

    def _make_shard_state(self, batch, output, tags, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "tags": tags,
        }

    def _compute_loss(self, batch, output, target, copy_attn, align, tags):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """

        # Copy alignment is tgt x batch x src
        src_len = copy_attn.shape[2]

        # Use supervision on the mask prediction
        tagging_loss = 0

        target = target.view(-1)
        align = align.view(-1)
        scores, gumbel_tags, extra_stats = self.generator(
            self._bottle(output),
            self._bottle(copy_attn),
            tags,
            batch.src_map)

        loss = self.criterion(scores, align, target)
        scores_data = scores.data.clone()
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores_data, batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[inputters.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).float().sum(0)
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        # penalize selection

        tagging_penalty = gumbel_tags.view(-1, 2)[:, 1].sum()
        extra_stats['tagging_penalty'] = [tagging_penalty.item() / copy_attn.shape[1]]
        stats = self._stats(loss_data, scores_data, target_data, extra_stats)


        if self.generator.annealing_steps > self.generator.start_annealing_steps:

            loss = loss + tagging_penalty * 0.0001

     #        print("Loss: {:.3f}, Penalty: {:.3f}".format(
        # loss.data.item(), tagging_penalty.item()))
            # for tag in gumbel_tags.view(-1, 2)[:15, 1]:
            #     print("{:.3f}".format(tag.item()), end=" ")
            # print()
        else:
            loss = loss + tagging_penalty * 0.

        return loss, stats
