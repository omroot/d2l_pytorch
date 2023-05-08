
import math
import pandas as pd
import torch
from torch import nn


from omd2l.models.Transformer.PositionWiseFFN import PositionWiseFFN
from omd2l.models.Transformer.AddNorm import AddNorm
from omd2l.models.AttentionSeq2Seq.MultiHeadAttention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self,
                 num_hiddens,
                 ffn_num_hiddens,
                 num_heads,
                 dropout,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens,
                                            num_heads,
                                            dropout,
                                            use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))