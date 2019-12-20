# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class EmbeddingNet(nn.Module):
    def __init__(self, image=False, audio=False, text=False, time=False,
                merge='concat'):
        super(EmbeddingNet, self).__init__()

        nn_input = 0
        self.image_feature = False
        self.audio_feature = False
        self.text_feature  = False
        self.timestamp     = False
        self.merge         = merge

        if image:
            self.image_feature = True 
            nn_input += 2048
        if audio:
            self.audio_feature = True 
            nn_input += 2560
        if text:
            self.text_feature = True
            nn_input += 768
        if time:
            self.timestamp = True
            nn_input += 3

        print('nn_input:', nn_input)

        # normal
        self.fc = nn.Sequential(nn.Linear(nn_input, 512), nn.ReLU(),
                                nn.Linear(512, 128), nn.ReLU(),
                                nn.Linear(128, 30))

        # with BatchNorm
        self.fc_bn = nn.Sequential(nn.Linear(nn_input, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                nn.Linear(256, 128))

        # with dropout
        self.fc_do = nn.Sequential(nn.Linear(nn_input, 512), nn.PReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(512, 256), nn.PReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(256, 128))
    
    def forward(self, x):
        concat_list = []

        if self.image_feature:
            img_feature = x['image']
            # print('img_feature: ', img_feature.size())
            concat_list.append(img_feature)
        if self.audio_feature:
            aud_feature = x['audio']
            # print('aud_feature: ', aud_feature.size())
            concat_list.append(aud_feature)
        if self.text_feature:
            txt_feature = x['text']
            # print('txt_feature: ', txt_feature.size())
            concat_list.append(txt_feature)
        if self.timestamp:
            concat_list.append(x['timestamp'])
        
        if self.merge == 'concat':
            output = torch.cat(concat_list, dim=1)

        print('output size' ,output.size()) # ([3, nn_input])
        output = self.fc(output)
        
        return output
    
    def get_embedding(self, x):
        return self.forward(x)
