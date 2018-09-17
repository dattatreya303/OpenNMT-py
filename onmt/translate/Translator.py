import argparse
import torch
import codecs
import os
import math

from torch.autograd import Variable
from itertools import count

import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts

import torchtext


def make_translator(opt, report_score=True, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, **kwargs)
    return translator

from copy import deepcopy
from collections import Counter


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

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 gpu=-1,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None):
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.model.eval()

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self, src_dir, src_path, tgt_path,
                  batch_size, attn_debug=False):
        data = onmt.io.build_dataset(self.fields,
                                     self.data_type,
                                     src_path,
                                     tgt_path,
                                     src_dir=src_dir,
                                     sample_rate=self.sample_rate,
                                     window_size=self.window_size,
                                     window_stride=self.window_stride,
                                     window=self.window,
                                     use_filter_pred=self.use_filter_pred)


        data_iter = onmt.io.OrderedIterator(
            dataset=data, device=self.gpu,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        for batch in data_iter:
            batch_data = self.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[0]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            self._report_score('PRED', pred_score_total, pred_words_total)
            if tgt_path is not None:
                self._report_score('GOLD', gold_score_total, gold_words_total)
                if self.report_bleu:
                    self._report_bleu(tgt_path)
                if self.report_rouge:
                    self._report_rouge(tgt_path)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores

    def translate_batch(self, batch, data, return_states=False, partial=[], attn_overwrite=[]):
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

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
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

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        # prepare for attention overwrite
        # attn_overwrite = [Counter(a) for a in attn_overwrite]
        words_so_far = 0
        # If we have partial translation, run decoder over them
        pref_attn = None
        if partial:
            print("partial in Translator", partial)
            partial_pre = [p[:-1] for p in partial]
            _, dec_states, __, pref_attn = self._run_pred(src, memory_bank,
                                                          enc_states, batch,
                                                          partial_pre)
            # Pref attn is word x batch x source
            # This I need to modify in beam
            for b, p in zip(beam, partial):
                b.next_ys[0][0] = p[-1]

            # Update counter of how many words were already seen
            words_so_far = len(partial_pre[0]) + 1
        else:
            dec_states = self.model.decoder.init_decoder_state(
                src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
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

            # Get the current attention overwrite
            new_attn = [a[words_so_far] if words_so_far in a.keys() else -1 for a in attn_overwrite]
            if all(a == -1 for a in new_attn):
                new_attn = []
            # Run one step.
            dec_out, dec_states, attn, weighted_context = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths,
                attn_overwrite=new_attn)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
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
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

            # Update seen words
            words_so_far += 1

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam, partial, pref_attn)

        # Compute the beam trace
        trace = {}
        # If we have prefix decoding, add this to beam
        if partial:
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
            if partial and trace[j]:
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
            ret["context"] = memory_bank
            """
            Predictions are a list per input
            Context is going round robin
            resorting....
            """
            resorted = []
            for topIx in range(self.n_best):
                cbatch = []
                for bIx in range(batch.batch_size):
                    cpred = ret["predictions"][bIx][topIx]
                    cbatch.append(cpred)

                resorted.append(cbatch)

            target_states = [[] for predIx in range(batch.batch_size)]
            target_context = [[] for predIx in range(batch.batch_size)]
            for b in resorted:
                tstates, _, cstar, attn = self._run_pred(src, memory_bank, enc_states,
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
            res = self._get_top_k(src, memory_bank, enc_states,
                                  batch, resorted[0], data=data)
            ret["beam"] = res
            ret["beam_trace"] = trace

            ret["target_states"] = target_states
            ret["target_cstar"] = target_context

            # Todo: add copy attn if applicable, add copy switch for each step
        return ret

    def _from_beam(self, beam, partial=[], pref_attn=None):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for j, b in enumerate(beam):
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            if partial:
                prefix = partial[j]
                prev_attn = pref_attn[:,j,:].data.squeeze(1)
            else:
                prefix = []
            for i, (times, k) in enumerate(ks[:n_best]):
                # print(times)
                # for ix in range(times):
                #     h, _ = b.get_hyp(ix, k)
                #     print(h)
                hyp, att = b.get_hyp(times, k)
                if partial:
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
        # Set copied OOV words to UNK
        if self.copy_attn:
            tgt_in = tgt_in.masked_fill(
                tgt_in.gt(len(self.fields["tgt"].vocab) - 1), 0)
        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)
        _, src_lengths = batch.src

        dec_out, dec_states, attn, weighted_context = self.model.decoder(
            tgt_in, context, dec_states, memory_lengths=src_lengths)
        # Special case -> only <s> gets fed
        try:
            dec_out_ret = dec_out[1:]
            weighted_context_ret = weighted_context[:, 1:]
        except:
            dec_out_ret = None
            weighted_context_ret = None

        return dec_out_ret, dec_states, weighted_context_ret, attn['std']

    def _get_top_k(self, 
                   src, 
                   context, 
                   enc_states, 
                   batch, 
                   pred, 
                   k=5,
                   data=None):
        """
        Computes the top k predictions for each time step.
        """
        tt = torch.cuda if self.cuda else torch
        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        if self.copy_attn:
            src_vocab = torchtext.vocab.Vocab(Counter([int(s) for s in src]),
                                              specials=[0, 
                                                        1])
            src_map = torch.LongTensor([src_vocab.stoi[int(w)] for w in src]).unsqueeze(1)
            src_size = len(src)
            #print(int(src.max()))
            src_vocab_size = int(src_map.max()) + 1
            # print(src_size, src_vocab_size)
            alignment = torch.zeros(src_size, 1, src_vocab_size)
            for j, t in enumerate(src_map):
                alignment[int(j), 0, int(t)] = 1
            src_map = Variable(alignment)
        else:
            src_map = None

        # print(src_map.shape)
        max_len = len(max(pred, key=len))
        context = context[:, :batch.batch_size, :]
        pred_in = []
        for p in pred:
            while len(p) < max_len:
                p.append(tgt_pad)
            p = [self.fields["tgt"].vocab.stoi[onmt.io.BOS_WORD]] + p[:-2]
            pred_in.append(tt.LongTensor(p).unsqueeze(1))
        tgt_in = Variable(torch.stack(pred_in, 1))
        # Set copied OOV words to UNK
        if self.copy_attn:
            tgt_in = tgt_in.masked_fill(
                tgt_in.gt(len(self.fields["tgt"].vocab) - 1), 0)


        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)
        _, src_lengths = batch.src

        dec_out, dec_states, attn, __ = self.model.decoder(
            tgt_in, context, dec_states, memory_lengths=src_lengths)
        print(dec_out.shape, attn['copy'].shape)
        alt_preds = []
        alt_scores = []
        if not self.copy_attn:
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
        else:
            for ix, (dec, copy_attn) in enumerate(zip(dec_out, attn['copy'])):
                out = self.model.generator.forward(dec,
                                                   copy_attn,
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    out.data.view(1, 1, -1),
                    batch, self.fields["tgt"].vocab, [src_vocab])
                #     # beam x tgt_vocab
                #     out = out.log()
                #     beam_attn = unbottle(attn["copy"])

                # each o is a different prediction
                alt = []
                alt_s = []
                for o in out:
                    o = o.view(-1)
                    best_scores, best_scores_id = o.squeeze().topk(k, 0, True, True)
                    alt.append(best_scores_id)
                    alt_s.append(best_scores)
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
                tgt_in[ix].unsqueeze(0), context, dec_states, memory_lengths=src_lengths)
            fix_dec_states = deepcopy(dec_states)
            c_out = []
            for ix2, a in enumerate(alt):
                # Forward the latest tokens
                if int(a) > len(self.fields["tgt"].vocab) -1:
                    a = a.fill_(0)
                d_out, d_states, _, __ = self.model.decoder(
                    Variable(a.unsqueeze(0)), context, fix_dec_states, memory_lengths=src_lengths)
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
                    current_dic = {"pred": int(pr),
                                   "score": float(sc),
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
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def _report_score(self, name, score_total, words_total):
        print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))

    def _report_bleu(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (path, tgt_path, self.output),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        print(">> " + res.strip())

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        print(res.strip())
