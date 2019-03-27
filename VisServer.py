from __future__ import division, unicode_literals
import argparse
import codecs
import sys

import h5py
import torch


import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules

from onmt.opts import model_opts, translate_opts


PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def traverse_reply(rep, depth=0):
    indent = "\t" * depth
    if type(rep) == dict:
        for key, value in rep.items():
            print(indent + str(key))
            traverse_reply(value, depth=depth+1)
    elif type(rep) == list:
        traverse_reply(rep[0], depth=depth+1)
    else:
        print(indent + str(type(rep)))


class ONMTmodelAPI():
    def __init__(self, model_loc, opt={'gpu': -1,
                                       'beam_size': 5,
                                       'n_best': 5,
                                       'alpha': 0,
                                       'beta': 0
                                       }):
        # Simulate all commandline args
        # (need to shut down the real argv for this)
        old_argv = sys.argv.copy()
        sys.argv = sys.argv[:1]
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
        sys.argv = old_argv
        # Model load options
        dummy_parser = argparse.ArgumentParser(description='train.py')
        model_opts(dummy_parser)
        self.dummy_opt = dummy_parser.parse_known_args([])[0]

        # Load the model.
        self.fields, self.model, self.model_opt = \
            onmt.model_builder.load_test_model(
                self.opt, self.dummy_opt.__dict__)

        # Make GPU decoding possible
        # self.opt.gpu = gpu
        self.opt.cuda = self.opt.gpu > -1
        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpu)

        # Translator
        self.scorer = onmt.translate.GNMTGlobalScorer(self.opt)

        self.translator = onmt.translate.Translator(
            self.model, self.fields,
            self.opt,
            self.model_opt,
            global_scorer=self.scorer)

    @staticmethod
    def default_inference_options():
        return {'k': 1,
                'beam_size': 10,
                'min_length': 35,
                'stepwise_penalty': True,
                'coverage_penalty': 'summary',
                'beta': 10,
                'length_penalty': 'wu',
                'alpha': 1.0,
                'block_ngram_repeat': 3,
                'ignore_when_blocking': [".", "</t>", "<t>"],
                'replace_unk': True,
                }

    def format_payload(self, translation_list, batch_data, in_text):
        """
        Structure of Payload

        OLD:

        {1: {
            # top_k
            scores: [top1_score, top2_score, ...]
            # src
            encoder: [{'token': str,
                       'state': [float, float, ...]},
                      {},
                      ...
                      ],
            # top_k x tgt_len
            decoder: [[{'token': str,
                       'state': [float, float, ...],
                       'cstar': [float, float, ...]},
                       {},
                       ...
                      ],
                      [],
                      ...
                     ]
            # top_k x max_tgt x src
            attn: [[[float, float, float],
                    [],
                    ...
                    ],
                    [],
                    ...
                   ]
            # max_tgt x beam_size
            beam: [[{'pred': int,
                     'score': float,
                     'state: [float, float, ...]'}
                   ],
                   [],
                   ...
                  ]
            # max_txt x beam_size x curr_step
            beam_trace: [[[int], [int], ...],
                         [[hyp, hyp], [hyp, hyp], ...],
                         ...
                        ]
            },
         2: { ...}
         }


        NEW:

        {1: {
            # top_k
            scores: [top1_score, top2_score, ...]
            # src
            encoder: [{'token': str,
                       'state': [float, float, ...],
                       'XXX': XXX},
                      {},
                      ...
                      ],
            # top_k x tgt_len
            decoder: [[{'token': str,
                       'state': [float, float, ...],
                       'context': [float, float, ...],
                        'XXX': XXX},
                       {},
                       ...
                      ],
                      [],
                      ...
                     ]
            # top_k x max_tgt x src
            alignment: {'attn': [[[float, float, float],
                                    [],
                                    ...
                                    ],
                                    [],
                                    ...
                                   ],
                        'XXX': XXX
                        }
            # max_tgt x beam_size
            beam: [[{'pred': str,
                     'score': float,
                     'state: [float, float, ...]'}
                   ],
                   [],
                   ...
                  ]
            # max_txt x beam_size x curr_step
            beam_trace: [[[str], [str], ...],
                         [[hyp, hyp], [hyp, hyp], ...],
                         ...
                        ]
            },
         2: { ...}
         }
        """

        # print(batch_data['target_extra'][0])
        # print(batch_data['target_context'])
        reply = {}
        for transIx, trans in enumerate(translation_list):
            res = {}
            # Fill encoder Result
            encoderRes = []
            context = batch_data['context'][:, transIx, :]
            for token, state in zip(in_text[transIx].split(), context):
                encoderRes.append({'token': token,
                                   'state': [0] # state.data.tolist()
                                   })
            res['encoder'] = encoderRes

            # Fill decoder+Attn Result
            decoderRes = []
            attnRes = []
            copyAttnRes = []
            for ix, p in enumerate(trans.pred_sents[:self.translator.n_best]):
                if not p:
                    continue
                topIx = []
                topIxAttn = []
                topIxCopyAttn = []
                for tokIx, (token, attn, copy_attn, state, cstar) in enumerate(zip(
                        p,
                        trans.attns[ix],
                        trans.copy_attns[ix],
                        batch_data["target_states"][transIx][ix],
                        batch_data['target_context'][transIx][ix])):
                    currentDec = {}
                    currentDec['token'] = token
                    currentDec['state'] = [0]   # state.data.tolist()
                    currentDec['context'] = [0] # cstar.data.tolist()
                    # Extra tgt annotations
                    for key, value in batch_data['target_extra'][transIx][ix].items():
                        currentDec[key] = float(value[tokIx+1].item())
                    topIx.append(currentDec)
                    topIxAttn.append(attn.tolist())
                    topIxCopyAttn.append(copy_attn.tolist())
                    # if t in ['.', '!', '?']:
                    #     break
                decoderRes.append(topIx)
                attnRes.append(topIxAttn)
                copyAttnRes.append(topIxCopyAttn)
            res['decoder'] = decoderRes
            res['attn'] = attnRes
            res['copy_attn'] = copyAttnRes

            pred_scores = [r.cpu().numpy().tolist() for r in
                           trans.pred_scores[:self.translator.n_best]]
            res['scores'] = pred_scores
            res['beam'] = batch_data['beam'][transIx]
            res['beam_trace'] = batch_data['beam_trace'][transIx]

            # Set reply index
            reply[transIx] = res
        return reply

    def dump_data(self):
        """
        Writes information from the model to files:
        - source and tgt dictionaries
        - Encoder and Decoder embeddings
        """
        with open("s2s/src.dict", 'w') as f:
            for w, ix in self.translator.fields['src'].vocab.stoi.items():
                f.write(str(ix) + " " + w + "\n")
        with open("s2s/tgt.dict", 'w') as f:
            for w, ix in self.translator.fields['tgt'].vocab.stoi.items():
                f.write(str(ix) + " " + w + "\n")
        with h5py.File("s2s/embs.h5", 'w') as f:
            f.create_dataset("encoder", data=self.translator.model.encoder.embeddings.emb_luts[0].weight.data.numpy())
            f.create_dataset("decoder", data=self.translator.model.decoder.embeddings.emb_luts[0].weight.data.numpy())

    def update_translator(self, options={'k': 5}):
        """
        Based on sent options, we can update the Scorer and other inference parameter
        """

        # set n_best in translator
        self.translator.n_best = options['k']

        # Increase Beam size if asked for large k
        if self.translator.beam_size < options['k']:
            self.translator.beam_size = options['k']

        # Overwrite Scorer params
        for key, item in options.items():
            setattr(self.opt, key, item)
        # Update max length of prediction
        self.translator.max_length = self.opt.max_length
        self.translator.stepwise_penalty = self.opt.stepwise_penalty
        self.translator.min_length = self.opt.min_length
        self.translator.block_ngram_repeat = self.opt.block_ngram_repeat
        self.translator.ignore_when_blocking = self.opt.ignore_when_blocking
        self.translator.max_sentences = self.opt.max_sentences

        # Set new Global Scorer
        self.scorer = onmt.translate.GNMTGlobalScorer(self.opt)
        self.translator.global_scorer = self.scorer

    def translate(self,
                  in_text,
                  partial_decode=[],
                  attn_overwrite=[],
                  inference_options={'k': 5},
                  dump_data=False,
                  selection_mask=None):
        """
        in_text: list of strings
        partial_decode: list of strings, not implemented yet
        attn_overwrite: dictionary of which index in decoder
                        has what attention on the encoder
        k: int, number of top translations to return
        attn: list, not implemented yet
        selection_mask: list of list of 0/1 for each input
        """

        # Set batch size to number of requested translations
        self.opt.batch_size = len(in_text)
        self.update_translator(inference_options)

        # Code to extract the source and target dict
        if dump_data:
            self.dump_data()

        # Write input to file for dataset builder
        with codecs.open("tmp.txt", "w", "utf-8") as f:
            for line in in_text:
                f.write(line + "\n")

        # Use written file as input to dataset builder
        data = onmt.inputters.build_dataset(
            self.fields,
            self.opt.data_type,
            "tmp.txt",
            tgt=self.opt.tgt,
            src_dir=self.opt.src_dir,
            sample_rate=self.opt.sample_rate,
            window_size=self.opt.window_size,
            window_stride=self.opt.window_stride,
            window=self.opt.window,
            use_filter_pred=False,
            dynamic_dict=self.model_opt.copy_attn)

        # Iterating over the single batch... torchtext requirement
        cur_device = "cuda" if self.opt.cuda else "cpu"
        test_data = onmt.inputters.OrderedIterator(
            dataset=data, device=torch.device(cur_device),
            batch_size=self.opt.batch_size, train=False, sort=False,
            sort_within_batch=True,
            shuffle=False)

        # Builder used to convert translation to text
        builder = onmt.translate.TranslationBuilder(
            data, self.translator.fields,
            self.translator.n_best, self.opt.replace_unk, self.opt.tgt)

        # Convert partial decode into valid input to decoder
        if partial_decode:
            print("partial:", partial_decode)
        vocab = self.fields["tgt"].vocab
        partial = []
        for p in partial_decode:
            curr_part = []
            for tok in p.split():
                curr_part.append(vocab.stoi[tok])
            partial.append(curr_part)

        # Retrieve batch to translate
        # We only have one batch, but indexing does not work
        for b in test_data:
            batch = b

        # Format the selection mask
        if selection_mask is not None:
            # Check correct length
            selection_mask = torch.FloatTensor(selection_mask)
            assert(batch.src[0].shape == selection_mask.transpose(0, 1).shape)

        # Run the translation
        batch_data = self.translator.translate_batch(
            batch, data, False,
            return_states=True,
            partial=partial,
            attn_overwrite=attn_overwrite,
            selection_mask=selection_mask)
        translations = builder.from_batch(batch_data)

        # Format to specified format
        payload = self.format_payload(
            translation_list=translations,
            batch_data=batch_data,
            in_text=in_text)

        # For debugging, uncomment this
        # traverse_reply(payload)
        return payload


def main():
    model = ONMTmodelAPI("model/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt")

    def print_only_pred_text(rep):
        '''
        Debug function
        '''
        print("Predictions:")
        for k in reply[0]['decoder']:
            for tok in k:
                print(tok['token'], end=" ")
            print("")

    # model = ONMTmodelAPI("../Seq2Seq-Vis/0316-fakedates/date_acc_100.00_ppl_1.00_e7.pt")
    # model = ONMTmodelAPI("models/ende_acc_46.86_ppl_21.19_e12.pt")

    # Summarization Inference options
    inference_options = {'k': 1,
                         'beam_size': 10,
                         'min_length': 5,
                         'stepwise_penalty': True,
                         'coverage_penalty': 'summary',
                         'beta': 10,
                         'length_penalty': 'wu',
                         'alpha': 1.0,
                         'block_ngram_repeat': 3,
                         'ignore_when_blocking': [".", "</t>", "<t>"],
                         'replace_unk': True,
                         'max_sentences': 6
                         }

    # Simple Case
    reply = model.translate(["this is a test ubiquotus ."],
                            dump_data=False,
                            inference_options=inference_options)
    print_only_pred_text(reply)
    # Selection Mask
    reply = model.translate(["this is a test ubiquotus ."],
                            selection_mask=[[1, 1, 1, 1, 0, 1]],
                            inference_options=inference_options)
    print_only_pred_text(reply)
    # Prefix
    reply = model.translate(["this is a long test ubiquotus ."],
                            inference_options=inference_options,
                            partial_decode=["<t> this is"])
    print_only_pred_text(reply)

    # Prefix and Mask
    # reply = model.translate(["this is a test ubiquotus ."],
    #                         selection_mask=[[1, 1, 1, 1, 0, 1]],
    #                         inference_options=inference_options,
    #                         partial_decode=["<t> this is"])
    # print_only_pred_text(reply)

    # reply = model.translate(["scientists at nasa are one step closer to understanding how much water could have existed on primeval mars . these new findings also indicate how primitive water reservoirs there could have evolved over billions of years , indicating that early oceans on the red planet might have held more water than earth 's arctic ocean , nasa scientists reveal in a study published friday in the journal science . `` our study provides a solid estimate of how much water mars once had , by determining how much water was lost to space , '' said geronimo villanueva , a scientist at nasa 's goddard space flight center . `` with this work , we can better understand the history of water on mars . '' to find answers to this age-old question about martian water molecules , scientists used the world 's three major infrared telescopes , in chile and hawaii , to measure traces of water in the planet 's atmosphere over a range of areas and seasons , spanning from march 2008 to january 2014 . `` from the ground , we could take a snapshot of the whole hemisphere on a single night , '' said goddard 's michael mumma . scientists looked at the ratio of two different forms -- or isotopes -- of water , h2o and hdo . the latter is made heavier by one of its hydrogen atoms , called deuterium , which has a neutron at its core in addition to the proton that all hydrogen atoms have . that weighed down hdo more , while larger amounts of hydrogen from h2o floated into the atmosphere , broke away from mars ' low gravity and disappeared into space . as a result , water trapped in mars ' polar ice caps has a much higher level of hdo than fluid water on earth does , the scientists said . the scientists compared the ratio of h2o to hdo in mars ' atmosphere today to the ratio of the two molecules trapped inside a mars meteorite , a stone that broke off from mars -- perhaps when an asteroid hit -- and landed on earth some 4.5 billion years ago . they were able to determine how much that ratio had changed over time and estimate how much water has disappeared from mars -- about 87 % . the findings indicate that the red planet could have had its fair share of blue waters , possibly even yielding an ocean . according to nasa , there might have been enough water to cover up to 20 % of mars ' surface . that would amount to an ocean proportionally larger than the atlantic on earth . `` this ocean had a maximum depth of around 5,000 feet or around one mile deep , '' said villanueva . nasa scientists say that much of this water loss happened over billions of years , along with a loss of atmosphere . and as the planet 's atmospheric pressure dropped , it was harder for water to stay in liquid form . heat also contributed to its evaporation . as a result , the remaining primeval ocean water continued to move toward the poles , where it eventually froze . `` with mars losing that much water , the planet was very likely wet for a longer period of time than was previously thought , suggesting it might have been habitable for longer , '' said mumma . cnn 's ben brumfield contributed to this report ."],
    #                         inference_options=inference_options,
    #                         partial_decode=[" ".join(['<t>', 'nasa', 'scientists', 'say', 'that', 'much', 'of', 'this', 'water', 'loss', 'happened', 'over', 'billions', 'of', 'years', ',', 'along', 'with', 'a', 'loss', 'of', 'atmosphere', '.', '</t>'])])
    # print_only_pred_text(reply)

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

    # print_only_pred_text(reply)
    # print(json.dumps(reply, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
