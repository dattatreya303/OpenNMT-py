import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io

from copy import deepcopy


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        self.model.eval()

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data, return_states=False, partial=None):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           return_states: whether to return states as well
           partial: partial input to the decoder


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, context = self.model.encoder(src, src_lengths)
        # If we have partial translation, run decoder over them
        if partial:
            print("partial in Translator", partial)
            partial_pre = [p[:-1] for p in partial]
            _, dec_states, __, pref_attn = self._run_pred(src, context,
                                                          enc_states, batch,
                                                          partial_pre)
            # Pref attn is word x batch x source
            # This I need to modify in beam
            for b, p in zip(beam, partial):
                b.next_ys[0][0] = p[-1]
        else:
            dec_states = self.model.decoder.init_decoder_state(
                src, context, enc_states)


        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn, weighted_context = self.model.decoder(
                inp, context, dec_states, context_lengths=context_lengths)

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :context_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam, partial, pref_attn)

        # Compute the beam trace
        trace = {}
        # If we have prefix decoding, add this to beam
        if partial is not None:
            for ix in range(len(partial)):
                all_current = []
                last = []
                # Ignore last index, since that is on the beam
                for wIx in partial[ix][:-1]:
                    last.append(wIx)
                    # print("last", last)
                    all_current.append([last.copy()] * self.beam_size)
                    # print("ac", all_current)
                trace[ix] = all_current
        for j, b in enumerate(beam):
            # k holds the chosen beam, y the predictions
            if partial:
                all_current = trace[j]
                last = trace[j][-1]
            else:
                all_current = []
                last = []

            for ix, cur_tops in enumerate(b.prev_ks):
                # print("cur_top", cur_tops.numpy())
                # print("cur_preds", b.next_ys[ix].numpy())
                if not last:
                    last = [[b.next_ys[0][k]] for k in cur_tops]
                else:
                    last = [last[b.prev_ks[ix][a]] + [b.next_ys[ix][k]] for a, k
                            in enumerate(cur_tops)]
                all_current.append(last)
            trace[j] = all_current

        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch

        if return_states:
            ret["context"] = context
            """
            Predictions are a list per input
            Context is going round robin
            resorting....
            """
            resorted = []
            for topIx in range(self.n_best):
                cbatch = []
                for bIx in range(batch.batch_size):
                    if partial is not None:
                        cpred = partial[bIx] +  ret["predictions"][bIx][topIx]
                    else:
                        cpred = ret["predictions"][bIx][topIx]
                    cbatch.append(cpred)

                resorted.append(cbatch)

            target_states = [[] for predIx in range(batch.batch_size)]
            target_context = [[] for predIx in range(batch.batch_size)]
            for b in resorted:
                tstates, _, cstar, attn = self._run_pred(src, context, enc_states,
                                           batch, b)
                tstates = tstates.squeeze()
                cstar = cstar.squeeze()

                if batch.batch_size > 1:
                    for predIx in range(batch.batch_size):
                        target_states[predIx].append(tstates[:,predIx,:].squeeze())
                        target_context[predIx].append(cstar[predIx,:,:].squeeze())
                else:
                    target_states[0].append(tstates)
                    target_context[0].append(cstar)
            # Get the top 5 for each time step
            res = self._get_top_k(src, context, enc_states,
                                  batch, resorted[0])
            ret["beam"] = res
            ret["beam_trace"] = trace

            ret["target_states"] = target_states
            ret["target_cstar"] = target_context
        return ret

    def _from_beam(self, beam, partial=None, pref_attn=None):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for j, b in enumerate(beam):
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            if partial is not None:
                prefix = partial[j]
                prev_attn = pref_attn[:,j,:].data.squeeze()
            else:
                prefix = []
            for i, (times, k) in enumerate(ks[:n_best]):
                # print(times)
                # for ix in range(times):
                #     h, _ = b.get_hyp(ix, k)
                #     print(h)
                hyp, att = b.get_hyp(times, k)
                if partial is not None:
                    src_width = att.size(1)
                    att = torch.cat([prev_attn[:,:src_width], att], dim=0)
                hyps.append(prefix + hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        # print(ret)
        return ret

    def _run_pred(self, src, context, enc_states, batch, pred):
        tt = torch.cuda if self.cuda else torch
        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        max_len = len(max(pred, key=len))
        context = context[:, :batch.batch_size, :]
        pred_in = []
        for p in pred:
            while len(p) < max_len:
                p.append(tgt_pad)
            p = [self.fields["tgt"].vocab.stoi[onmt.io.BOS_WORD]] + p
            pred_in.append(tt.LongTensor(p).unsqueeze(1))
        tgt_in = Variable(torch.stack(pred_in, 1))

        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)
        _, src_lengths = batch.src

        dec_out, dec_states, attn, weighted_context = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)
        return dec_out[1:], dec_states, weighted_context[:,1:], attn['std']

    def _get_top_k(self, src, context, enc_states, batch, pred, k=5):
        tt = torch.cuda if self.cuda else torch
        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        max_len = len(max(pred, key=len))
        context = context[:, :batch.batch_size, :]
        pred_in = []
        for p in pred:
            while len(p) < max_len:
                p.append(tgt_pad)
            p = [self.fields["tgt"].vocab.stoi[onmt.io.BOS_WORD]] + p[:-2]
            pred_in.append(tt.LongTensor(p).unsqueeze(1))
        tgt_in = Variable(torch.stack(pred_in, 1))
        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)
        _, src_lengths = batch.src

        dec_out, dec_states, _, __ = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)

        alt_preds = []
        alt_scores = []
        for ix, dec in enumerate(dec_out):
            out = self.model.generator.forward(dec)
            # each o is a different prediction
            alt = []
            alt_s = []
            for o in out:
                o = o.view(-1)
                best_scores, best_scores_id = o.squeeze().topk(k, 0, True, True)
                alt.append(best_scores_id.data)
                alt_s.append(best_scores.data)

                # print(best_scores.data, best_scores_id.data)
            alt_preds.append(alt)
            alt_scores.append(alt_s)

        # Construct new inputs
        out_states = []
        dec_states = self.model.decoder.init_decoder_state(
                    src, context, enc_states)
        for ix, t in enumerate(tgt_in.data):
            alt = torch.stack(alt_preds[ix]).view(k, -1).unsqueeze(2)
            prev = tgt_in.data[:ix+1]
            # Precompute initial state
            dec_out, dec_states, _, __ = self.model.decoder(
                tgt_in[ix].unsqueeze(0), context, dec_states, context_lengths=src_lengths)
            fix_dec_states = deepcopy(dec_states)
            c_out = []
            for ix2, a in enumerate(alt):
                # Forward the latest tokens
                d_out, d_states, _, __ = self.model.decoder(
                    Variable(a.unsqueeze(0)), context, fix_dec_states, context_lengths=src_lengths)
                c_out.append(d_out.data)
                # write these into the correct positions d_out -> in decoder states one per step
            out_states.append(c_out)


        # Assemble reply
        res = {ix:[] for ix in range(tgt_in.size(1))}
        # Iterate over time steps
        for targ, stat, pred, sco in zip(tgt_in, out_states, alt_preds, alt_scores):

            # Iterate over batch
            for ix, t in enumerate(targ):
                outs = []
                # ignore padding in here

                if t.data[0] == tgt_pad \
                   or t.data[0] == self.fields["tgt"].vocab.stoi[onmt.io.EOS_WORD] \
                   or t.data[0] == self.fields["tgt"].vocab.stoi["."]:
                    continue
                current_state = [list(s[:,ix,:].squeeze().numpy()) for s in stat]
                for pr, sc, st in zip(list(pred[ix].numpy()), list(sco[ix].numpy()), current_state):
                    current_dic = {"pred": pr,
                                   "score": sc,
                                   "state": st}
                    outs.append(current_dic)

                res[ix].append(outs)
        return res

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
