# coding:utf-8
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, csv_path=None, transform=None, train=True, image=True, audio=False):
        """
        1. Image
        2. TimeStamp( start_sec, end_sec, shot_sec)
        3. Audio
        """
        self.train = train # これでtrain/testの切り替えを行う
        self.image_load = image
        self.audio_load = audio 
        self.train_df = pd.read_csv('./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv')
        self.transform = transform

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
        # print(self.train_df.head())
        print('start_sec: ', len(self.start_sec))
        print('end_sec: ', len(self.end_sec))
        print('shot_sec', len(self.shot_sec))
        print('images', len(self.images))
        print('labels', len(self.labels))
        print('============================')
    
    def image_open(self, t):
        image = Image.open(t)
        # transformするならする
        if self.transform is None:
            return image
        else:
            return self.transform(image)
    
    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0,2) # 0or1をランダムに選択
            img1, label1 = self.images[index], self.scene_id[index]
            timestamp1 = [self.start_sec[index], self.end_sec[index], self.shot_sec[index]]

            label_count = len(self.label_to_indices[label1])

            # negative pairs
            # 同一ショットが一つの場合もtarget=0に
            if target == 0 or label_count < 2 :
                print('target == 0')
                target = 0 # 強制的にtargetを0に
                siamese_label = np.random.choice(list(self.labels_set - (set([label1]))))
                siamese_index =  np.random.choice(self.label_to_indices[siamese_label])
            # positive pairs
            else:
                print('target == 1')
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])

            img2 = self.images[siamese_index]
            timestamp2 = [self.start_sec[siamese_index], self.end_sec[siamese_index], self.shot_sec[siamese_index]]
        else:
            # TODO:テストデータ用の処理を書く
            pass

        # img1 = self.image_open(img1)
        # img2 = self.image_open(img2)
        
        return (img1,timestamp1,img2,timestamp2), target

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])

    multimodaldataset = MultiModalDataset(transform=transform ,train=True)
