# coding:utf-8
import numpy as np
import pandas as pd
import librosa, scipy
import os, sys, glob, csv
from sklearn.preprocessing import StandardScaler
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms

class SiameseMulti(Dataset):
    def __init__(self, test_path='./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv',
                    train=True, image=False, timestamp=False, audio=False, text=False, weight='place'):
        """
        1. Image: ImageNet or Place-365 (weight)
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

            scaler = StandardScaler()
            self.start_sec = list(scaler.fit_transform(self.train_df.start_sec))
            self.end_sec   = list(scaler.fit_transform(self.train_df.end_sec))
            self.shot_sec  = list(scaler.fit_transform(self.train_df.shot_sec))

            if self.image_load: 
                if weight == 'place':
                    self.images = list(self.train_df.place365_feature_path)
                else:
                    self.images = list(self.train_df.imagenet_feature_path)
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
    def __init__(self, test_path='./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv',
                    train=True, image=False, timestamp=False, audio=False, text=False, weight='place'):
        """
        train: 各サンプル(アンカー)に対して、正と負のサンプルをランダムに選択する
        """
        self.train          = train # これでtrain/testの切り替えを行う
        self.image_load     = image
        self.timestamp_load = timestamp
        self.audio_load     = audio 
        self.text_load      = text 
    
        """ train data merge and load """
        if self.train:
            """ test以外のcsvファイルをすべてマージしてtrainとする """
            train_csv_list = list(set(glob.glob(os.path.dirname(test_path)+'/*')) - set([test_path]))

            self.train_df = None
            for trian_csv in train_csv_list:
                if self.train_df is None:
                    self.train_df = pd.read_csv(trian_csv)
                else:
                    _df = pd.read_csv(trian_csv)
                    shot_id = self.train_df['shot_id'].max()+1
                    scene_id = self.train_df['scene_id'].max()+1
                    _df['shot_id'] = _df['shot_id'] + shot_id
                    _df['scene_id'] = _df['scene_id'] + scene_id
                    self.train_df = pd.concat([self.train_df, _df])

            """ loading """
            self.index         = list(self.train_df.shot_id)
            self.labels     = list(self.train_df.scene_id)
            self.labels_set = set(self.train_df.scene_id.unique())
            self.label_to_indices = {label: np.where(self.train_df.scene_id == label)[0]
                                    for label in self.labels_set}
            # print('self.label_set:', self.labels_set)
            # print('self.labels_to_indices:',  self.label_to_indices)
            scaler = StandardScaler()
            self.start_sec = list(scaler.fit_transform(self.train_df.start_sec))
            self.end_sec   = list(scaler.fit_transform(self.train_df.end_sec))
            self.shot_sec  = list(scaler.fit_transform(self.train_df.shot_sec))

            if self.image_load: 
                if weight == 'place':
                    self.images = list(self.train_df.place365_feature_path)
                else:
                    self.images = list(self.train_df.imagenet_feature_path)
            if self.audio_load: self.audios = list(self.train_df.audio_feature_path)
            if self.text_load:  self.texts  = list(self.train_df.text_feature_path)
        else: # setup test data
            self.test_df = pd.read(test_path)
            self.test_labels = list(self.test_df.scene_id)
            self.labels_set = set(self.test_df.scene_id.unique())
            self.label_to_indices = {label: np.where(self.test_df.scene_id == label)[0]
                                    for label in self.labels_set}
            print('self.labels_set:', self.labels_set)
            print('self.labels_to_indices:', self.label_to_indices)
            self.start_sec = list(self.train_df.start_sec)
            self.end_sec   = list(self.train_df.end_sec)
            self.shot_sec  = list(self.train_df.shot_sec)

            if self.image_load: 
                self.images = list(self.test_df.image_feature_path)
                self.images_path = list(self.test_df.image)
            if self.audio_load: self.audios = list(self.test_df.audio_feature_path)
            if self.text_load:  self.texts  = list(self.test_df.text_feature_path)

            random_state = np.random.RandomState(77)

            # 無限ループの可能性アリ
            triplets = [[i,
                        random_state.choice(self.label_to_indices[self.test_labels[i]]),
                        random_state.choice(self.label_to_indices[
                                                random.choice(
                                                    list(self.labels_set - set([self.test_labels[i]]))
                                                )
                                            ])
            ]
            for i in range(len(self.test_df))]
            self.test_triplets = triplets

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
            img_anc = self.load_image(anchor_index)
            img_pos = self.load_image(positive_index)
            img_neg = self.load_image(negative_index)
            anchor['image']   = img_anc
            positive['image'] = img_pos
            negative['image'] = img_neg
        
        """ load audio """
        if self.audio_load:
            aud_anc = self.load_audio(anchor_index)
            aud_pos = self.load_audio(positive_index)
            aud_neg = self.load_audio(negative_index)
            anchor['audio'] = aud_anc
            positive['audio'] = aud_pos
            negative['audio'] = aud_neg
        
        """ load text """
        if self.text_load:
            txt_anc = self.text_load(anchor_index)
            txt_pos = self.text_load(positive_index)
            txt_neg = self.text_load(negative_index)
            anchor['text'] = txt_anc
            positive['text'] = txt_pos
            negative['text'] = txt_neg
        
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
    
    def load_image(self, index):
        img = np.load(self.images[index])
        img = torch.from_numpy(img)
        img = torch.squeeze(img, dim=0)
        return img

    def load_audio(self, index):
        aud = np.load(self.audios[index])
        aud = torch.from_numpy(aud)
        aud = torch.squeeze(adr.view(aud.size()[0], -1), dim=0)
        return aud

    def load_text(self, index):
        txt = np.load(self.texts[index])
        txt = torch.from_numpy(txt)
        return txt

    def __len__(self):
        return len(self.labels) if self.train else len(self.test_labels)

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])

    multimodaldataset = SiameseMulti(train=False, image=False, audio=True, timestamp=True, text=True)
