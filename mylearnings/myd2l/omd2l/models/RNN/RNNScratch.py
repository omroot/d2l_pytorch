




import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import gym
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from scipy.spatial import distance_matrix


import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


from omd2l.models.base.Module import Module
from omd2l.models.optimizer.SGD import SGD
import omd2l.utils.compute  as compute

class RNNScratch(Module):

    def __init__(self,
                 num_inputs: int,
                 num_hiddens: int,
                 sigma: float=0.01):
        super().__init__()
        self.save_hyperparameters()
        # weight parameters applied to the input from the current time step
        self.W_xh = nn.Parameter(
            compute.randn(num_inputs, num_hiddens) * sigma)
        # weight parameters applied to the previous time step
        self.W_hh = nn.Parameter(
            compute.randn(num_hiddens, num_hiddens) * sigma)
        # Bias parameter
        self.b_h = nn.Parameter(compute.zeros(num_hiddens))

    def forward(self, inputs, state=None):

        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = compute.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        else:
            state, = state
        outputs = []
        # Iterate over the sequence
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = compute.tanh(compute.matmul(X, self.W_xh) +
                             compute.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state


