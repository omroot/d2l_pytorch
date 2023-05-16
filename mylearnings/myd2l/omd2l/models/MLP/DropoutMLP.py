





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

from omd2l.models.base.Classifier import Classifier
from omd2l.models.optimizer.SGD import SGD
import omd2l.utils.compute  as compute


class DropoutMLP(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
                            nn.Flatten(),
                            nn.LazyLinear(num_hiddens_1),
                            nn.ReLU(),
                            nn.Dropout(dropout_1),
                            nn.LazyLinear(num_hiddens_2),
                            nn.ReLU(),
                            nn.Dropout(dropout_2),
                            nn.LazyLinear(num_outputs))