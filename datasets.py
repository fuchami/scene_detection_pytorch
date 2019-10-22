# coding:utf-8
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, transform=None, train=True, image=True, audio=False):
        """
        """
        self.train = train # これでtrain/testの切り替えを行う
        self.image = image
        self.audio = audio 
        self.train_df = pd.read_csv('./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv')
        self.transform = transform

        self.labels = list(self.train_df.scene_id.unique())
        self.labels_set = set(self.train_df.scene_id.unique())
        self.label_to_indices = {label: np.where(self.train_df.scene_id == label)[0]
                                for label in self.labels_set}
        print('self.label_set:', self.labels_set)
        print('self.labels_to_indices:',  self.label_to_indices)
        if self.image: self.images = list(self.train_df.image.unique())
        if self.audio: self.audios = list(self.train_df.audio.unique())

        print(self.train_df.head())
        print(self.labels)
    
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
            img1, label1 = self.images[index], self.labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - (set([label1]))))
                siamese_index =  np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_df[siamese_index]
        else:
            # TODO:テストデータ用の処理を書く
            pass

        img1 = self.image_open(img1)
        img2 = self.image_open(img2)
        
        return (img1, img2), target

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