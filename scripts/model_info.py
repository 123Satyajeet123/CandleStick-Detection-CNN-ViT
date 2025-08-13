# load the best cnn model and check model summary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
import torchvision.io as io
import torchvision.io.image as image

# load the best cnn model
model = torch.load('models/cnn/best_model.pth')
