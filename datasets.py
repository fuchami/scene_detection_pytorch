# coding:utf-8
import numpy as np
import pandas as pd
import librosa, scipy
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms

class SiameseMulti(Dataset):
    def __init__(self, csv_path=None, transform=None, train=True,
                    image=False, timestamp=False, audio=False, text=False):
        """
        1. Image
        2. TimeStamp( start_sec, end_sec, shot_sec)
        3. Audio
        4. text
        """
        self.train_df       = pd.read_csv('./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv')
        self.train          = train # これでtrain/testの切り替えを行う
        self.transform      = transform
        self.image_load     = image
        self.timestamp_load = timestamp
        self.audio_load     = audio 
        self.text_load      = text 

        self.labels = list(self.train_df.scene_id)
        self.labels_set = set(self.train_df.scene_id.unique())
        self.label_to_indices = {label: np.where(self.train_df.scene_id == label)[0]
                                for label in self.labels_set}
        # print('self.label_set:', self.labels_set)
        # print('self.labels_to_indices:',  self.label_to_indices)
        self.start_sec = list(self.train_df.start_sec)
        self.end_sec   = list(self.train_df.end_sec)
        self.shot_sec  = list(self.train_df.shot_sec)

        if self.image_load: self.images = list(self.train_df.image.unique())
        if self.audio_load: self.audios = list(self.train_df.audio.unique())

        print('============================')
        print('--- MultimodalDataset ---')
        print(self.train_df.head())
        print('start_sec: ', len(self.start_sec))
        print('end_sec: ', len(self.end_sec))
        print('shot_sec', len(self.shot_sec))
        print('labels', len(self.labels))
        print('============================')
    
    def image_open(self, t):
        image = Image.open(t)
        # transformするならする
        if self.transform is None:
            return image
        else:
            return self.transform(image)
        
    def audio_open(self, t):
        y, fs = librosa.load(t)
        melsp = librosa.feature.melspectrogram(y=y, sr=fs)

        # (1, 128, 431)のtensorへ
        melsp = torch.unsqueeze(torch.tensor(melsp), 0)
        # print(melsp.size())
        return melsp
    
    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0,2) # 0or1をランダムに選択
            label1 = self.labels[index]

            if self.image_load: img1 = self.images[index]
            if self.audio_load: aud1 = self.audios[index]
            if self.timestamp_load: timestamp1 = [self.shot_sec[index]]

            label_count = len(self.label_to_indices[label1])

            # negative pairs
            # 同一ショットが一つの場合もtarget=0に
            if target == 0 or label_count < 2 :
                # print('target == 0')
                target = 0 # 強制的にtargetを0に
                siamese_label = np.random.choice(list(self.labels_set - (set([label1]))))
                siamese_index =  np.random.choice(self.label_to_indices[siamese_label])
            # positive pairs
            else:
                # print('target == 1')
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])

            if self.image_load: img2 = self.images[siamese_index]
            if self.audio_load: aud2 = self.audios[siamese_index]
            if self.timestamp_load: timestamp2 = [self.shot_sec[siamese_index]]

        else:
            # TODO:テストデータ用の処理を書く
            pass

        data1 = {}
        data2 = {}

        """ load image """
        if self.image_load:
            img1 = self.image_open(img1)
            img2 = self.image_open(img2)
            data1['image'] = img1
            data2['image'] = img2

        """ load audio """
        if self.audio_load:
            aud1 = self.audios[index]
            aud1 = self.audio_open(aud1)
            aud2 = self.audios[siamese_index]
            aud2 = self.audio_open(aud2)
            data1['audio'] = aud1
            data2['audio'] = aud2

        """ load timestamp """
        if self.timestamp_load:
            timestamp1 = torch.tensor(timestamp1)
            timestamp2 = torch.tensor(timestamp2)
            data1['timestamp'] = timestamp1
            data2['timestamp'] = timestamp2
        
        dataset = (data1, data2)
        return dataset, target, label1

    def __len__(self):
        return len(self.labels)

class TripletMulti(Dataset):
    """
    train: 各サンプル(アンカー)に対して、正と負のサンプルをランダムに選択する
    """

    def __init__(self, csv_path=None, transform=None, train=True,
                    image=False, timestamp=False, audio=False, text=False):

        self.train_df       = pd.read_csv('./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv')
        self.train          = train # これでtrain/testの切り替えを行う
        self.transform      = transform
        self.image_load     = image
        self.timestamp_load = timestamp
        self.audio_load     = audio 
        self.text_load      = text 

        self.labels = list(self.train_df.scene_id)
        self.labels_set = set(self.train_df.scene_id.unique())
        self.label_to_indices = {label: np.where(self.train_df.scene_id == label)[0]
                                for label in self.labels_set}
        # print('self.label_set:', self.labels_set)
        # print('self.labels_to_indices:',  self.label_to_indices)
        self.start_sec = list(self.train_df.start_sec)
        self.end_sec   = list(self.train_df.end_sec)
        self.shot_sec  = list(self.train_df.shot_sec)

        if self.image_load: self.images = list(self.train_df.image.unique())
        if self.audio_load: self.audios = list(self.train_df.audio.unique())

        print('============================')
        print('--- Triplet MultimodalDataset ---')
        print(self.train_df.head())
        print('start_sec: ', len(self.start_sec))
        print('end_sec: ', len(self.end_sec))
        print('shot_sec', len(self.shot_sec))
        print('labels', len(self.labels))
        print('============================')
        print(self.label_to_indices)
        print('============================')
    
    def image_open(self, t):
        image = Image.open(t)
        # transformするならする
        if self.transform is None:
            return image
        else:
            return self.transform(image)
        
    def audio_open(self, t):
        y, fs = librosa.load(t)
        melsp = librosa.feature.melspectrogram(y=y, sr=fs)

        # (1, 128, 431)のtensorへ
        melsp = torch.unsqueeze(torch.tensor(melsp), 0)
        # print(melsp.size())
        return melsp
    
    def __getitem__(self, index):
        return 


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])

    multimodaldataset = TripletMulti(transform=transform ,train=True,
                                            image=False, audio=True, timestamp=False)
