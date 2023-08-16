

from torch import nn
nn_Module = nn.Module
import inspect
class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k :v for k, v in local_vars.items()
                        if k not in set(ignore +['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
