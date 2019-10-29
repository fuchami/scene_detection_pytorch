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
        print('audio_cnn forward:',x.size())

        return x

class SiameseNet(nn.Module):
    def __init__(self, image_extractor, image=False, audio=False, timestamp=False):
        super(SiameseNet, self).__init__()
        self.image_extractor = image_extractor
        self.image_extractor.eval() # 評価用に固定すべき?
        self.audio_feature = True if audio else False
        self.image_feature = True if image else False
        self.timestamp = True if timestamp else False

        nn_input = 0
        if self.audio_feature: nn_input += 2048
        if self.image_feature: nn_input += 2048
        if self.timestamp: nn_input += 1

        print('nn_input:', nn_input)

        self.fc = nn.Sequential(nn.Linear(nn_input, 512), nn.PReLU(), nn.Dropout(0.5), # 画像特徴+Timestamp
                                nn.Linear(512, 128), nn.PReLU(), nn.Dropout(0.5),
                                nn.Linear(128, 2))
        
        """ build audio cnn """
        if self.audio_feature: self.audio_cnn = AudioCNN()
    
    def forward(self, audio1, audio2):
        """
        output1 = self.image_extractor(img1)
        output1 = torch.cat([output1,timestamp1], dim=1)
        output1 = self.fc(output1)

        output2 = self.image_extractor(img2)
        output2 = torch.cat([output2,timestamp2], dim=1)
        output2 = self.fc(output2)
        """

        output1 = self.audio_cnn(audio1)
        print('output1:', output1.size())
        output1 = self.fc(output1)

        output2 = self.audio_cnn(audio2)
        output2 = self.fc(output2)

        return output1, output2

    def get_embedding(self, audio):
        """
        x = self.image_extractor(img)
        x = torch.cat([x, timestamp], dim=1)
        x =  self.fc(x)
        """
        x = self.audio_cnn(audio)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    audiocnn = AudioCNN()
    print(audiocnn)

    x = torch.rand(1,1,128, 431)
    out = audiocnn(x)


    """
    image_extractor = ImageExtractor()
    siamesenet = SiameseNet(image_extractor)
    print(siamesenet)
    """