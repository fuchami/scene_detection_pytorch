# coding:utf-8
import numpy as np
import pandas as pd
import librosa, scipy
import os, sys, glob, csv
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms

class SiameseMulti(Dataset):
    def __init__(self, test_path='./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv',
                    train=True, image=False, timestamp=False, audio=False, text=False):
        """
        1. Image: ResNet or VGG 
        2. Audio: VGGish 
        4. text: BERT 
        4. TimeStamp( start_sec, end_sec, shot_sec)
        """
        self.train          = train # これでtrain/testの切り替えを行う
        self.image_load     = image
        self.timestamp_load = timestamp
        self.audio_load     = audio 
        self.text_load      = text 

        """ train data loading... """
        if self.train:
            """ test以外のcsvファイルをすべてマージしてtrainとする"""
            train_csv_list = list(set(glob.glob(os.path.dirname(test_path)+'/*')) - set([test_path]))
            # print('train data list:', train_csv_list)

            self.train_df = None
            for train_csv in train_csv_list:
                if self.train_df is None:
                    self.train_df = pd.read_csv(train_csv)
                else:
                    _df = pd.read_csv(train_csv)
                    shot_id = self.train_df['shot_id'].max() +1
                    scene_id = self.train_df['scene_id'].max() +1
                    _df['shot_id'] = _df['shot_id'] + shot_id
                    _df['scene_id'] = _df['scene_id'] + scene_id

                    self.train_df = pd.concat([self.train_df, _df])

            """ loading """
            self.labels = list(self.train_df.scene_id)
            self.labels_set = set(self.train_df.scene_id.unique())
            self.label_to_indices = {label: np.where(self.train_df.scene_id == label)[0]
                                    for label in self.labels_set}
            # print('self.label_set:', self.labels_set)
            # print('self.labels_to_indices:',  self.label_to_indices)
            self.start_sec = list(self.train_df.start_sec)
            self.end_sec   = list(self.train_df.end_sec)
            self.shot_sec  = list(self.train_df.shot_sec)

            if self.image_load: self.images = list(self.train_df.image_feature_path)
            if self.audio_load: self.audios = list(self.train_df.audio_feature_path)
            if self.text_load:  self.texts  = list(self.train_df.text_feature_path)

        else:
            self.test_df = pd.read_csv(test_path)
            self.test_labels = list(self.test_df.scene_id)
            self.labels_set  = set(self.test_df.scene_id.unique())
            self.label_to_indices = {label: np.where(self.test_df.scene_id == label)[0]
                                    for label in self.labels_set}
            # print('self.label_set:', self.labels_set)
            # print('self.label_to_indices:', self.label_to_indices)
            self.start_sec = list(self.test_df.start_sec)
            self.end_sec   = list(self.test_df.end_sec)
            self.shot_sec  = list(self.test_df.shot_sec)

            if self.image_load:
                self.images = list(self.test_df.image_feature_path)
                self.images_path = list(self.test_df.image)
            if self.audio_load: self.audios = list(self.test_df.audio_feature_path)
            if self.text_load:  self.texts  = list(self.test_df.text_feature_path)
            
            random_state = np.random.RandomState(77)

            positive_pairs = [[i,
                                random_state.choice(self.label_to_indices[self.test_labels[i]]),
                                1]
                                for i in range(0, len(self.test_df),2)]
            negative_pairs = [[i,
                                random_state.choice(self.label_to_indices[
                                                        np.random.choice(
                                                            list(self.labels_set - set([self.test_labels[i]]))
                                                        )
                                                    ]),
                                0]
                                for i in range(0, len(self.test_df),2)]
            
            self.test_pairs = positive_pairs + negative_pairs
    
    def __getitem__(self, index):
        data1 = {}
        data2 = {}

        if self.train:
            target = np.random.randint(0,2) # 0or1をランダムに選択
            label1 = self.labels[index]
            img1_path = ""

            label_count = len(self.label_to_indices[label1])

            # negative pairs
            # 同一ショットが一つの場合もtarget=0に
            if target == 0 or label_count < 2 :
                target = 0 # 強制的にtargetを0に
                siamese_label = np.random.choice(list(self.labels_set - (set([label1]))))
                siamese_index =  np.random.choice(self.label_to_indices[siamese_label])
            # positive pairs
            else:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])

        else:
            index = self.test_pairs[index][0]
            siamese_index = self.test_pairs[index][1]
            target = self.test_pairs[index][2]
            label1 = self.test_labels[index]
            img1_path = self.images_path[index]

        if self.audio_load: aud1 = self.audios[index]
        if self.text_load:  txt1 = self.texts[index]
        if self.audio_load: aud2 = self.audios[siamese_index]
        if self.text_load:  txt2 = self.texts[siamese_index]

        """ load image """
        if self.image_load:
            img1 = np.load(self.images[index])
            img2 = np.load(self.images[siamese_index])
            img1 = torch.from_numpy(img1) # torch.Size([1, 2048])
            img2 = torch.from_numpy(img2)

            img1 = torch.squeeze(img1, dim=0)
            img2 = torch.squeeze(img2, dim=0)  
            data1['image'] = img1
            data2['image'] = img2

        """ load audio """
        if self.audio_load:
            aud1 = np.load(self.audios[index])
            aud2 = np.load(self.audios[siamese_index])
            aud1 = torch.from_numpy(aud1) # torch.Size([1, 20, 128])
            aud2 = torch.from_numpy(aud2)

            aud1 = torch.squeeze(aud1.view(aud1.size()[0], -1), dim=0)
            aud2 = torch.squeeze(aud2.view(aud2.size()[0], -1), dim=0)
            data1['audio'] = aud1
            data2['audio'] = aud2
        
        """ load text """
        if self.text_load:
            txt1 = np.load(self.texts[index])
            txt2 = np.load(self.texts[siamese_index])
            txt1 = torch.from_numpy(txt1) # torch.Size([768])
            txt2 = torch.from_numpy(txt2)
            data1['text'] = txt1
            data2['text'] = txt2

        """ load timestamp """
        if self.timestamp_load:
            """ time stamp 1 feature or 3 feature? """
            # if self.timestamp_load: timestamp1 = [self.shot_sec[index]]
            # if self.timestamp_load: timestamp2 = [self.shot_sec[siamese_index]]
            if self.timestamp_load: timestamp1 = [self.shot_sec[index], self.start_sec[index], self.end_sec[index]]
            if self.timestamp_load: timestamp2 = [self.shot_sec[siamese_index], self.start_sec[siamese_index], self.end_sec[siamese_index]]

            timestamp1 = torch.tensor(timestamp1)
            timestamp2 = torch.tensor(timestamp2)
            data1['timestamp'] = timestamp1
            data2['timestamp'] = timestamp2
        
        dataset = (data1, data2)
        return dataset, target, label1, img1_path

    def __len__(self):
        return len(self.labels) if self.train else len(self.test_labels)

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

        self.index         = list(self.train_df.shot_id)
        self.labels     = list(self.train_df.scene_id)
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
            anchor_index = index
            anchor_label = self.labels[anchor_index]
            label_count = len(self.label_to_indices[anchor_label])
            img1_path = self.images[index]


            # 2つ以上のショットのシーンが出るまでがんばる
            while label_count < 2:
                anchor_index = np.random.choice(self.index)
                anchor_label = self.labels[anchor_index]
                label_count = len(self.label_to_indices[anchor_label])
            
            positive_index = anchor_index
            while positive_index == anchor_index:
                positive_index = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        else:
            pass # TODO: テストデータ用の処理を書く
            

        """ label check """
        if self.labels[anchor_index] != self.labels[positive_index]:
            raise ("ERROR!!! anchor and positive not same ")
        if self.labels[anchor_index] == self.labels[negative_index]:
            raise ("ERROR!!! anchor and negative same ")
        if self.labels[positive_index] == self.labels[negative_index]:
            raise ("ERROR!!! positive and negative same ")

        anchor = {}
        positive = {}
        negative = {}

        """ load image """
        if self.image_load:
            img_anc = self.image_open(self.images[anchor_index])
            img_pos = self.image_open(self.images[positive_index])
            img_neg = self.image_open(self.images[negative_index])
            anchor['image']   = img_anc
            positive['image'] = img_pos
            negative['image'] = img_neg
        
        """ load audio """
        if self.audio_load:
            aud_anc = self.audio_open(self.audios[anchor_index])
            aud_pos = self.audio_open(self.audios[positive_index])
            aud_neg = self.audio_open(self.audios[negative_index])
            anchor['audio'] = aud_anc
            positive['audio'] = aud_pos
            negative['audio'] = aud_neg
        
        """ laod timestamp """
        if self.timestamp_load:
            ts_anc = [self.shot_sec[anchor_index], self.start_sec[anchor_index], self.end_sec[anchor_index]]
            ts_pos = [self.shot_sec[positive_index], self.start_sec[positive_index], self.end_sec[positive_index]]
            ts_neg = [self.shot_sec[negative_index], self.start_sec[negative_index], self.end_sec[negative_index]]
            anchor['timestamp'] = torch.tensor(ts_anc)
            positive['timestamp'] = torch.tensor(ts_pos)
            negative['timestamp'] = torch.tensor(ts_neg)

        dataset = (anchor, positive, negative)
        return dataset,[], anchor_label, img1_path

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])

    multimodaldataset = SiameseMulti(train=False, image=False, audio=True, timestamp=True, text=True)
