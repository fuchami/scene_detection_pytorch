# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.embedding import EmbeddingNet

# from torchsummary import summary

class SiameseNet(nn.Module):
    def __init__(self, image=False, audio=False, text=False, time=False):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(image,audio,text,time)
    
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)

        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    def __init__(self, image=False, audio=False, text=False, time=False):
        super(TripletNet, self).__init__()
        self.embedding_net = EmbeddingNet(image,audio,text,time)
    
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)

        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
