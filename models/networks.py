# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from torchsummary import summary

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
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # self.fc = nn.Linear()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        print(x.size())
        return

class SiameseNet(nn.Module):
    def __init__(self, image_extractor, audio=False):
        super(SiameseNet, self).__init__()
        self.image_extractor = image_extractor
        self.image_extractor.eval() # 評価用に固定すべき?
        self.audio_feature = True if audio else False

        self.fc = nn.Sequential(nn.Linear(2048+1, 512), nn.PReLU(), nn.Dropout(0.5), # 画像特徴+Timestamp
                                nn.Linear(512, 128), nn.PReLU(), nn.Dropout(0.5),
                                nn.Linear(128, 2))
        
        """ build audio cnn """
        if self.audio_feature: audio_cnn = AudioCNN()
    
    def forward(self, img1, timestamp1, img2, timestamp2):
        output1 = self.image_extractor(img1)
        output1 = torch.cat([output1,timestamp1], dim=1)
        output1 = self.fc(output1)

        output2 = self.image_extractor(img2)
        output2 = torch.cat([output2,timestamp2], dim=1)
        output2 = self.fc(output2)

        return output1, output2

    def get_embedding(self, img, timestamp):
        x = self.image_extractor(img)
        x = torch.cat([x, timestamp], dim=1)
        x =  self.fc(x)
        return x

if __name__ == "__main__":
    audiocnn = AudioCNN()
    print(audiocnn)

    x = torch.rand(128, 431)
    out = audiocnn(x)
    print('out size:', out.size())



    """
    image_extractor = ImageExtractor()
    siamesenet = SiameseNet(image_extractor)
    print(siamesenet)
    """