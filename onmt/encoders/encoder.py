"""Base class for encoders and generic multi encoders."""
import torch

import torch.nn as nn

from onmt.utils.misc import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        raise NotImplementedError

    def _check_args(self, src, lengths=None, hidden=None):
        n_batch = src.size(1)
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``


        Returns:
            (FloatTensor, FloatTensor):

            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
        """

        raise NotImplementedError


class SiameseEncoder(EncoderBase):

    def __init__(self, k, m, n, batch_size):
        super(SiameseEncoder, self).__init__()
        self.batch_size = batch_size
        self.K = k
        self.M = m
        self.N = n
        self.W1 = nn.Linear(self.M, self.K)
        self.W2 = nn.Linear(self.K, self.N)
        self.src_indicator = torch.Tensor(batch_size, self.N, 1)
        self.doc_indicator = torch.Tensor(batch_size, self.M, 1)

    def from_opt(cls, opt, embeddings=None):
        pass

    def forward(self, src, lengths=None, src_doc_index=None, other_src_doc_index=None, vocab=None):

        # src_indicator = self._get_src_indicator(src)
        # v1 = torch.bmm(self.W2(self.W1(self._get_doc_indicator(src_doc_index))), src_indicator)
        # v2 = torch.bmm(self.W2(self.W1(self._get_doc_indicator(other_src_doc_index))), src_indicator)
        # return v1 - v2

        if other_src_doc_index == -1:
            other_src_doc_index = src_doc_index
        src_attn = self.W2(self.W1(self._get_doc_indicator(src_doc_index)))
        if other_src_doc_index is None:
            return src_attn, None
        other_src_attn = self.W2(self.W1(self._get_doc_indicator(other_src_doc_index)))
        return src_attn, other_src_attn

    def _get_doc_indicator(self, doc_index):
        return doc_index.view(doc_index.size()[1], doc_index.size()[0], doc_index.size()[2])

    def _get_src_indicator(self, src):
        self.src_indicator[:] = 0
        for bid in range(src.size(1)):
            for sid in range(src.size(0)):
                self.src_indicator[bid, src[sid, bid, 0], 0] = 1
        return self.src_indicator

    # tensor([33, 43, 1410, 6, 1096, 5])
