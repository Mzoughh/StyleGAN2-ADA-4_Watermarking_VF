# utils.py
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.autograd import Variable, grad
import torch.nn.functional as F
from collections import defaultdict


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
