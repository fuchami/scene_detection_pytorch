# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale # 係数weightの初期値として設定する値
        self.reset_parameters()
        self.eps = 1e-10
    
    def reset_parameters(self):
        """ 結合パラメータを大きさscaleの値にする初期化を実行 """
        nn.init.constant_(self.weight, self.scale) # weightの値がすべてscaleになる
    
    def forward(self, x):
        # norm tensor.size() ([batchsize, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out

class EmbeddingNet(nn.Module):
    def __init__(self, image=False, audio=False, text=False, time=False,
                merge='concat', outdim=128):
        super(EmbeddingNet, self).__init__()

        nn_input = 0
        self.image_feature = False
        self.audio_feature = False
        self.text_feature  = False
        self.timestamp     = False
        self.merge         = merge
        self.outdim        = outdim

        if self.merge == 'concat':
            if image:
                self.image_feature = True 
                nn_input += 2048
            if audio:
                self.audio_feature = True 
                nn_input += 1280
            if text:
                self.text_feature = True
                nn_input += 768
            if time:
                self.timestamp = True
                nn_input += 3

            print('nn_input:', nn_input)
        else:
            a_t_input = 768
            a_t_output = 2048
            nn_input = 4096

            self.fc_audio = nn.Sequential(nn.Linear(1280, 768), nn.BatchNorm1d(768), nn.PReLU())
            self.mcb_at = CompactBilinearPooling(a_t_input,a_t_input, a_t_output)
            self.mcb_it = CompactBilinearPooling(a_t_output,a_t_output, nn_input)

        # normal
        self.fc = nn.Sequential(nn.Linear(nn_input, 1024), nn.PReLU(),
                                nn.Linear(1024, 512), nn.PReLU(),
                                nn.Linear(512, 128))
        # with BatchNorm
        self.fc_bn = nn.Sequential(nn.Linear(nn_input, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
                                nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                nn.Linear(256, 128))
        # outdim32 with BatchNorm & dropout
        self.fc_bn_do = nn.Sequential(nn.Linear(nn_input, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                nn.Linear(512, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 32))

        # outdim128 with BatchNorm & dropout
        self.fc_bn_do2 = nn.Sequential(nn.Linear(nn_input, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 128))
    
    def forward(self, x):
        if self.merge == 'concat':
            concat_list = []

            if self.image_feature:
                img_feature = x['image']
                concat_list.append(img_feature)
            if self.audio_feature:
                aud_feature = x['audio']
                concat_list.append(aud_feature)
            if self.text_feature:
                txt_feature = x['text']
                concat_list.append(txt_feature)
            if self.timestamp:
                concat_list.append(x['timestamp'])
            output = torch.cat(concat_list, dim=1)
        else:
            audio_feature = self.fc_audio(x['audio'])
            at_output = self.mcb_at(x['text'], audio_feature)
            output = self.mcb_it(at_output, x['image'])

            output = torch.cat([output, x['timestamp']], dim=1)

        # print('output size' ,output.size()) # ([3, nn_input])
        if self.outdim == 128:
            output = self.fc_bn_do2(output)
        elif self.outdim == 32:
            output = self.fc_bn_do(output)
        else:
            raise ("ERROR!!! please select output dimention !!")
        
        return output
    
    def get_embedding(self, x):
        return self.forward(x)
