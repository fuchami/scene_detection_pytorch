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
                nn_input += 128
            if text:
                self.text_feature = True
                nn_input += 768
            if time:
                self.timestamp = True
                nn_input += 3
            # normal
            self.fc = nn.Sequential(nn.Linear(nn_input, 1024), nn.PReLU(),
                                    nn.Linear(1024, 512), nn.PReLU(),
                                    nn.Linear(512, 128))
            # with BatchNorm
            self.fc_bn = nn.Sequential(nn.Linear(nn_input, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                    nn.Linear(256, 128))

            # outdim32 with BatchNorm & dropout BEST!
            self.fc_bn_do = nn.Sequential(nn.Linear(nn_input, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                    nn.Linear(512, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, 32))

            # outdim with BatchNorm & dropout
            self.fc_bn_do2 = nn.Sequential(nn.Linear(nn_input, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                    nn.Linear(512, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                    nn.Dropout(0.5),
                                nn.Linear(128, 64))

            print('concat nn_input:', nn_input)
        elif self.merge == 'max': # 760でmaxpool
            print('feature max pooling!!!')

            self.fc_img = nn.Sequential(nn.Linear(2048, 128), nn.BatchNorm1d(128), nn.PReLU())
            self.fc_txt = nn.Sequential(nn.Linear(768, 128), nn.BatchNorm1d(128), nn.PReLU())
            # self.fc_aud = nn.Sequential(nn.Linear(1280, 768), nn.BatchNorm1d(768), nn.PReLU())

            self.emb_fc = nn.Sequential(nn.Linear(128+3, 90), nn.BatchNorm1d(90), nn.PReLU(),
                                nn.Linear(90, 64), nn.BatchNorm1d(64), nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, 32))
            return
        else: # 新しいmcb(image + text) + maxpool
            print('mcb + maxpooling!!!')
            mcb_input = 768
            mcb_output = 128

            nn_input = mcb_output+3 # add timestamp
            print('mcb nn_input:', nn_input)

            self.fc_img = nn.Sequential(nn.Linear(2048, 768), nn.BatchNorm1d(768), nn.PReLU())
            self.mcb = CompactBilinearPooling(mcb_input, mcb_input, mcb_output)

            self.emb_fc = nn.Sequential(nn.Linear(128+3, 64), nn.BatchNorm1d(64), nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, 32))
            return
    
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
            if self.outdim == 32:
                output = self.fc_bn_do(output)
            elif self.outdim == 64:
                output = self.fc_bn_do2(output)
            else:
                raise ("ERROR!!! please select output dimention !!")

        elif self.merge == 'max':
            """ max pooling """
            img_feature = self.fc_img(x['image'])
            txt_feature = self.fc_txt(x['text'])
            # aud_feature = self.fc_aud(x['audio'])

            txt_aud_max = torch.max(txt_feature, x['audio'])
            #print('debug:', txt_aud_max.size())
            img_txt_aud_max = torch.max(img_feature, txt_aud_max)
            # print('debug2:', img_txt_aud_max.size())

            output = torch.cat([img_txt_aud_max, x['timestamp']], dim=1)
            output = self.emb_fc(output)
        else:
            # image + mcb(text+aud) + timestamp
            img_feature = self.fc_img(x['image'])
            mcb_out = self.mcb(img_feature, x['text'])

            # mcb + audio maxpool
            output = torch.max(mcb_out, x['audio'])
            output = torch.cat([output, x['timestamp']], dim=1)
            output = self.emb_fc(output)
        
        return output
    
    def get_embedding(self, x):
        return self.forward(x)
