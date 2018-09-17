from __future__ import division, unicode_literals
import argparse
import codecs
import json
import sys

import h5py
import torch

import numpy as np
import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules

from onmt.opts import model_opts, translate_opts


PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'




def traverse_reply(rep, depth=0):
          indent = "\t" *depth
          if type(rep) == dict:
            for key, value in rep.items():
                print(indent + str(key))
                traverse_reply(value, depth=depth+1)
          elif type(rep) == list:
            traverse_reply(rep[0], depth=depth+1)
          else:
            print(indent + str(type(rep)))


class ONMTmodelAPI():
    def __init__(self, model_loc, opt={'gpu':-1,
                                       'beam_size': 5,
                                       'n_best': 5,
                                       }):
        # Simulate all commandline args
        parser = argparse.ArgumentParser(
            description='translate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        translate_opts(parser)
        
        # Add cmd opts (can also be used for other opts in future)
        opt['model'] = model_loc
        opt['src'] = "dummy_src"
        for (k, v) in opt.items():
            sys.argv += ['-%s' % k, str(v)]
        self.opt = parser.parse_args()

        # Model load options
        dummy_parser = argparse.ArgumentParser(description='train.py')
        model_opts(dummy_parser)
        self.dummy_opt = dummy_parser.parse_known_args([])[0]

        # Load the model.
        self.fields, self.model, self.model_opt = \
            onmt.ModelConstructor.load_test_model(
                self.opt, self.dummy_opt.__dict__)

        # Make GPU decoding possible
        # self.opt.gpu = gpu
        self.opt.cuda = self.opt.gpu > -1
        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpu)

        # Translator
        # TODO: expose the inference penalties
        self.scorer = onmt.translate.GNMTGlobalScorer(
            self.opt.alpha,
            self.opt.beta,
            cov_penalty=None, 
            length_penalty=None)

        self.translator = onmt.translate.Translator(
            self.model, self.fields,
            beam_size=self.opt.beam_size,
            n_best=self.opt.n_best,
            global_scorer=self.scorer,
            max_length=self.opt.max_length,
            copy_attn=self.model_opt.copy_attn,
            gpu=self.opt.gpu)


    def translate(self, in_text, partial_decode=[], attn_overwrite=[], k=5, attn=None, dump_data=False):
        """
        in_text: list of strings
        partial_decode: list of strings, not implemented yet
        k: int, number of top translations to return
        attn: list, not implemented yet
        """

        # Set batch size to number of requested translations
        self.opt.batch_size = len(in_text)
        # Workaround until we have API that does not require files
        with codecs.open("tmp.txt", "w", "utf-8") as f:
            for line in in_text:
                f.write(line + "\n")

        # Code to extract the source and target dict
        if dump_data:
            with open("s2s/src.dict", 'w') as f:
                for w, ix in self.translator.fields['src'].vocab.stoi.items():
                    f.write(str(ix) + " " + w + "\n")
            with open("s2s/tgt.dict", 'w') as f:
                for w, ix in self.translator.fields['tgt'].vocab.stoi.items():
                    f.write(str(ix) + " " + w + "\n")
            with h5py.File("s2s/embs.h5", 'w') as f:
                f.create_dataset("encoder", data=self.translator.model.encoder.embeddings.emb_luts[0].weight.data.numpy())
                f.create_dataset("decoder", data=self.translator.model.decoder.embeddings.emb_luts[0].weight.data.numpy())

        # Use written file as input to dataset builder
        data = onmt.io.build_dataset(
            self.fields, self.opt.data_type,
            "tmp.txt", self.opt.tgt,
            src_dir=self.opt.src_dir,
            sample_rate=self.opt.sample_rate,
            window_size=self.opt.window_size,
            window_stride=self.opt.window_stride,
            window=self.opt.window,
            use_filter_pred=False)

        # Iterating over the single batch... torchtext requirement
        test_data = onmt.io.OrderedIterator(
            dataset=data, device=self.opt.gpu,
            batch_size=self.opt.batch_size, train=False, sort=False,
            sort_within_batch=True,
            shuffle=False)

        # set n_best in translator
        self.translator.n_best = k

        # Increase Beam size if asked for large k
        if self.translator.beam_size < k:
            self.translator.beam_size = k

        # Builder used to convert translation to text
        builder = onmt.translate.TranslationBuilder(
            data, self.translator.fields,
            self.opt.n_best, self.opt.replace_unk, self.opt.tgt)

        # Convert partial decode into valid input to decoder
        print("partial:", partial_decode)
        vocab = self.fields["tgt"].vocab
        partial = []
        for p in partial_decode:
            curr_part = []
            for tok in p.split():
                curr_part.append(vocab.stoi[tok])
            partial.append(curr_part)

        reply = {}

        # Only has one batch, but indexing does not work
        for batch in test_data:
            batch_data = self.translator.translate_batch(
                batch, data, return_states=True,
                partial=partial, attn_overwrite=attn_overwrite)
            translations = builder.from_batch(batch_data)
            # iteratres over items in batch
            for transIx, trans in enumerate(translations):
                context = batch_data['context'][:, transIx, :]
                res = {}
                # Fill encoder Result
                encoderRes = []
                for token, state in zip(in_text[transIx].split(), context):
                    encoderRes.append({'token': token,
                                       'state': state.data.tolist()
                                       })
                res['encoder'] = encoderRes

                # # Fill decoder Result
                decoderRes = []
                attnRes = []
                for ix, p in enumerate(trans.pred_sents[:k]):
                    if p:
                        topIx = []
                        topIxAttn = []
                        for token, attn, state, cstar in zip(p,
                                                             trans.attns[ix],
                                                             batch_data["target_states"][transIx][ix],
                                                             batch_data['target_cstar'][transIx][ix]):
                            currentDec = {}
                            currentDec['token'] = token
                            currentDec['state'] = state.data.tolist()
                            currentDec['cstar'] = cstar.data.tolist()
                            topIx.append(currentDec)
                            topIxAttn.append(attn.tolist())
                            # if t in ['.', '!', '?']:
                            #     break
                        decoderRes.append(topIx)
                        attnRes.append(topIxAttn)
                res['scores'] = np.array(trans.pred_scores).tolist()[:k]
                res['decoder'] = decoderRes
                res['attn'] = attnRes
                res['beam'] = batch_data['beam'][transIx]
                res['beam_trace'] = batch_data['beam_trace'][transIx]
                reply[transIx] = res

        # For debugging, uncomment this
        # print(traverse_reply(reply))
        return json.dumps(reply)


def main():
    model = ONMTmodelAPI("models/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt")
    # model = ONMTmodelAPI("models/ende_acc_46.86_ppl_21.19_e12.pt")
    # Simple Case
    reply = model.translate(["this is a test ."], dump_data=True)
    # Case with attn overwrite OR partial
    # reply = model.translate(["this is madness ."], attn_overwrite=[{2:0}])
    # reply = model.translate(["this is madness ."], partial_decode=["das ist"])
    # Complex Case with attn and partial
    # reply = model.translate(["this is madness ."],
    #                         attn_overwrite=[{2:0}],
    #                         partial_decode=["das ist"])

    # Cases with multiple
    # reply = model.translate(["This is a test .", "and another one ."])
    # Partial
    # reply = model.translate(["This is a test .", "this is a second test ."],
    #                          partial_decode=["Dies ist", "Ein zweiter"])
    # Attn overwrite
    # reply = model.translate(["this is madness .", "i am awesome ."],
    #                         attn_overwrite=[{2:0}, {}])
    # All together - phew
    # reply = model.translate(["this is madness .", "i am awesome ."],
    #                         partial_decode=["heute ist", "du bist"],
    #                         attn_overwrite=[{2:0}, {2:2}])

    # Debug options
    # print("______")
    # print(len(reply[0]['decoder']))
    # print(len(reply[0]['decoder'][0]))
    # print(reply[0]['beam_trace'])
    # print(json.dumps(reply, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
