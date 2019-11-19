# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class EmbeddingNet(nn.Module):
    def __init__(self, image=False, audio=False, text=False, time=False,
                mcb=False):
        super(EmbeddingNet, self).__init__()

        nn_input = 0
        self.image_extract = False
        self.audio_extract = False
        self.text_extract  = False
        self.timestamp     = False
        self.mcb = mcb

        if image:
            self.image_extract = True 
            self.image_extractor = ImageExtractor()
            self.image_extractor.eval() # 評価用に固定すべき?
            nn_input += 2048
        if audio:
            self.audio_extract = True 
            self.audio_extractor = AudioCNN()
            nn_input += 2048
        if time:
            self.timestamp = True
            nn_input += 3

        print('nn_input:', nn_input)
        self.fc = nn.Sequential(nn.Linear(nn_input, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
                                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.PReLU(),
                                nn.Linear(512, 256))
    
    def forward(self, x):
        concat_list = []

        if self.image_extract:
            img_out = self.image_extractor(x['image'])
            concat_list.append(img_out)
        if self.audio_extract:
            aud_out = self.audio_extractor(x['audio'])
            concat_list.append(aud_out)
        if self.text_extract:
            pass
            # TODO: テキストの埋め込み処理
        if self.timestamp:
            concat_list.append(x['timestamp'])
        
        output = torch.cat(concat_list, dim=1)
        output = self.fc(output)
        
        return output
    
    def get_embedding(self, x):
        return self.forward(x)


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
            self.trained_model = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze layer
        for param in self.trained_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.trained_model(x)
        x = x.view(x.size()[0], -1)
        return x

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(256*9*28, 2048)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (1,1))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        # print('audio_cnn forward:',x.size())

        return x