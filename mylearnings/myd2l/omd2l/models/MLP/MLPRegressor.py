


from torch import nn


nn_Module = nn.Module


from omd2l.models.base.Regressor import Regressor



class MLPRegressor(Regressor):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LazyLinear(num_outputs))