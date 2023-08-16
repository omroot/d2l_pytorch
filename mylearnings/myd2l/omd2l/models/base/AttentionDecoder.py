


from torch import nn
nn_Module = nn.Module
from omd2l.models.base  import Decoder




class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError