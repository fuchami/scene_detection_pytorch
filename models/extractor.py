# coding:utf-8
import os,sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import BertModel, BertTokenizer
from torchvggish import vggish, vggish_input

class ImageExtractor(nn.Module):
    def __init__(self, model='res', weight='imagenet'):
        """
        pre-trained weight: iamgenet or place365
        """
        super(ImageExtractor, self).__init__()
        if weight=='imagenet':
            """ load ResNet-152 """
            resnet = models.resnet152(pretrained=True)
            self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        elif weight=='place365':
            """ load ResNet-50 """
            arch = 'resnet50'
            model_file = './models/resnet50_places365.pth.tar'
            if not os.access(model_file, os.W_OK):
                weight_url = 'https://places2.csail.mit.edu/models_places365' + model_file
                os.system('wget ' + weight_url)

            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            self.extractor = nn.Sequential(*list(model.children())[:-1])
        else:
            print('Error please select pre-trained weight: imagenet or place365')
            sys.exit(1)

    def forward(self, x):
        with torch.no_grad():
            x = self.extractor(x)

        # print('image extract size: ', x.size()) # [bs, 2048, 1, 1]
        x = x.view(x.size()[0], -1)
        print('image extract size: ', x.size()) # [bs, 2048]
        return x