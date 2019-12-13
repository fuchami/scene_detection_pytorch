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
            print(self.extractor)

        # only eval 
        self.extractor.eval()
    
    def forward(self, x):
        x = self.extractor(x)

        print('image extract size: ', x.size()) # [bs, 2048, 1, 1]
        x = x.view(x.size()[0], -1)
        print('image extract size: ', x.size()) # [bs, 2048]
        return x

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.bert.eval()
    
    def forward(self, txt):
        print('bert')
        txt = self.bert(txt)[0][0]
        txt = txt.view(txt.size()[0], -1)
        txt = torch.mean(txt, dim=2)
        return txt

class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.extractor = vggish()
        self.extractor.eval()
        # self.extractor.cuda()
    
    def forward(self, x):
        # x = vggish_input.wavfile_to_examples(x)
        x =self.extractor.forward(x)

        x = x.view(x.size()[0], -1)
        print('audio extract size: ', x.size()) # [bs, 2048]
        return x

