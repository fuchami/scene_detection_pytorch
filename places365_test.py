# coding:utf-8

#%%
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
import os
from PIL import Image

#%% load pre-trained weights
arch = 'resnet50'
model_file = 'resnet50_places365.pth.tar'
if not os.access(model_file, os.W_OK):
    weight_url = 'https://places2.csail.mit.edu/models_places365' + model_file
    os.system('wget ' + weight_url)

# %%
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
print(model)
model.eval()

# %%
resnet = nn.Sequential(*list(model.children())[:-1])
print(resnet)
# %%
