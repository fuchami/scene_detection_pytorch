# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchsummary import summary

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

class SiameseNet(nn.Module):
    def __init__(self, image_extractor):
        super(SiameseNet, self).__init__()
        self.image_extractor = image_extractor
        self.image_extractor.eval() # 評価用に固定すべき?

        self.fc = nn.Sequential(nn.Linear(2048+3, 512), nn.PReLU(), # 画像特徴+Timestamp
                                nn.Linear(512, 30), nn.PReLU(),
                                nn.Linear(30, 1))
    
    def forward(self, img1, timestamp1, img2, timestamp2):
        output1 = self.image_extractor(img1)
        output1 = torch.cat([output1,timestamp1], dim=1)
        output1 = self.fc(output1)

        output2 = self.image_extractor(img2)
        output2 = torch.cat([output2,timestamp2], dim=1)
        output2 = self.fc(output2)

        return output1, output2

    def get_embedding(self, x):
        return self.image_extractor(x)

if __name__ == "__main__":

    image_extractor = ImageExtractor()
    siamesenet = SiameseNet(image_extractor)
    print(siamesenet)