
from torch import nn


from omd2l.models.Transformer import PositionWiseFFN
from omd2l.models.Transformer import AddNorm
from omd2l.models.Transformer import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
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