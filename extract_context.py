#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import re
from itertools import chain

import torch

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts
import h5py
import numpy as np

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.translate_opts(parser)
opt = parser.parse_args()

dummy_parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(dummy_parser)
dummy_opt = dummy_parser.parse_known_args([])[0]

opt.cuda = opt.gpu > -1
if opt.cuda:
    torch.cuda.set_device(opt.gpu)

if torch.cuda.is_available() and not opt.gpu:
    print("WARNING: You have a CUDA device, should run with -gpu 0")


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            return sofar + max(len(new.tgt), len(new.src)) + 1

    device = opt.gpu if opt.gpu else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def extend_set(setname, bsize):
    setname.resize(setname.shape[0] + bsize, axis=0)


def extract_states(model, fields, data_type, model_opt, data_iter):
    """
    Writes the weighted contexts and attentions of the data set to h5
    """
    model.eval()
    print("Starting Extraction!")
    # train_datasets = lazily_load_dataset("train")
    train_iter = data_iter

    # src_vocab = fields['src'].vocab
    # tgt_vocab = fields['tgt'].vocab

    # Select max length to prune
    max_src_len = 75
    max_tgt_len = 75
    # Set path and remove if file exists already
    path = "S2S/states.h5"
    if os.path.isfile(path):
        os.remove(path)
    with h5py.File(path, "a") as f:
        # TODO increase chunk size
        srcset = f.create_dataset("src", (opt.batch_size,
                                          max_src_len),
                                  maxshape=(None, None),
                                  dtype="int",
                                  chunks=(10, 5))
        tgtset = f.create_dataset("tgt", (opt.batch_size,
                                          max_tgt_len),
                                  maxshape=(None, None),
                                  dtype="int",
                                  chunks=(10, 5))
        attnset = f.create_dataset("attn", (opt.batch_size,
                                            max_tgt_len,
                                            max_src_len),
                                   maxshape=(None, None, None),
                                   dtype="float16",
                                   chunks=(10, 5, 5))
        cstarset = f.create_dataset("cstar", (opt.batch_size,
                                              max_tgt_len,
                                              500),
                                    maxshape=(None, None, None),
                                    dtype="float16",
                                    chunks=(10, 5, 5))

        # Save encoder and decoder_hidden
        encoderset = f.create_dataset("encoder_out",
                                      (opt.batch_size,
                                       max_tgt_len,
                                       500),
                                      maxshape=(None, None, None),
                                      dtype="float16",
                                      chunks=(10, 5, 5))
        decoderset = f.create_dataset("decoder_out",
                                      (opt.batch_size,
                                       max_tgt_len,
                                       500),
                                      maxshape=(None, None, None),
                                      dtype="float16",
                                      chunks=(10, 5, 5))

        bcounter = 0
        for batch in train_iter:
            if bcounter > 0:
                extend_set(srcset, opt.batch_size)
                extend_set(tgtset, opt.batch_size)
                extend_set(attnset, opt.batch_size)
                extend_set(cstarset, opt.batch_size)
                extend_set(encoderset, opt.batch_size)
                extend_set(decoderset, opt.batch_size)

            # cur_dataset = train_iter.get_cur_dataset()
            src = onmt.io.make_features(batch, 'src', data_type)
            if data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _, weighted_context, context = model(src, tgt, src_lengths)

            # Get the sizes
            tgtsize, bsize, srcsize = attns['std'].size()

            # Pad and store the src
            padded_src = batch.src[0].data.numpy().transpose()
            padded_src = np.pad(padded_src,
                                ((0, 0),
                                 (0, max_src_len - srcsize)),
                                mode='constant',
                                constant_values=1)
            srcset[bcounter:] = padded_src

            # Pad and store the tgt
            # print("tgt", batch.tgt[:-1].size())
            padded_tgt = batch.tgt[:-1].data.numpy().transpose()
            padded_tgt = np.pad(padded_tgt,
                                ((0, 0),
                                 (0, max_tgt_len - tgtsize)),
                                mode='constant',
                                constant_values=1)
            tgtset[bcounter:] = padded_tgt

            # Pad and store the attention
            # print(attns['std'].size())
            # print(attns['std'].data[0])
            # print(torch.sum(attns['std'].data[0]))
            padded_attn = attns['std'].data.numpy()
            padded_attn = padded_attn.transpose((1,0,2))
            # print(padded_attn.shape)
            # print(padded_attn[:,0,:])
            padded_attn = np.pad(padded_attn,
                                 ((0, 0),
                                  (0, max_tgt_len - padded_attn.shape[1]),
                                  (0, max_src_len - padded_attn.shape[2])),
                                 mode='constant',
                                 constant_values=0.)
            attnset[bcounter:] = padded_attn

            # Pad and store the weighted context
            padded_c = weighted_context.data.numpy()
            padded_c = np.pad(padded_c,
                              ((0, 0),
                               (0, max_tgt_len - padded_c.shape[1]),
                               (0, 0)),
                              mode='constant',
                              constant_values=0.)
            cstarset[bcounter:] = padded_c

            # Pad and store decoder outputs (before generator)
            padded_enc = context.data.numpy()
            padded_enc = padded_enc.transpose((1,0,2))
            padded_enc = np.pad(padded_enc,
                              ((0, 0),
                               (0, max_src_len - padded_enc.shape[1]),
                               (0, 0)),
                              mode='constant',
                              constant_values=0.)
            encoderset[bcounter:] = padded_enc

            padded_dec = outputs.data.numpy()
            padded_dec = padded_dec.transpose((1,0,2))
            padded_dec = np.pad(padded_dec,
                              ((0, 0),
                               (0, max_tgt_len - padded_dec.shape[1]),
                               (0, 0)),
                              mode='constant',
                              constant_values=0.)
            decoderset[bcounter:] = padded_dec

            bcounter += bsize
            if bcounter % 100 == 0:
                print("Example #", bcounter)
            if bcounter > 100:
                break


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    print(model)

    return model


def main():
    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    data = onmt.io.build_dataset(fields, "text",
                                 opt.src, opt.tgt,
                                 use_filter_pred=False)
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    tally_parameters(model)

    # Extract the states.
    extract_states(model, fields, "text", model_opt, data_iter)


if __name__ == "__main__":
    main()
