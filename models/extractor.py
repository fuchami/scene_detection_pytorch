# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import BertModel, BertTokenizer
from torchvggish import vggish, vggish_input

class ImageExtractor(nn.Module):
    def __init__(self, model='res'):
        """
        res: ResNet-152
        vgg: VGG-16
        """
        super(ImageExtractor, self).__init__()
        if model=='res':
            """ load ResNet-152 """
            resnet = models.resnet152(pretrained=True)
            self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        if model=='vgg':
            vgg = models.vgg16(pretrained=True)
            self.extractor = nn.Sequential(*list(vgg.children())[:0])
            # print(self.extractor)

        # only eval 
        # self.extractor.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.extractor(x)

        # print('image extract size: ', x.size()) # [bs, 2048, 1, 1]
        x = x.view(x.size()[0], -1)
        print('image extract size: ', x.size()) # [bs, 2048]
        return x