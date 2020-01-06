# coding:utf-8

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

import torch
from torch.utils.data import Dataset

# cosine similarity
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# euclidean distance
def eucli_dis(v1, v2):
    return np.linalg.norm(v1-v2)

class PredData(Dataset):
    def __init__(self, test_path='./BBC_Planet_Earth_Dataset/dataset/annotator_0/01_From_Pole_to_Pole.csv',
                    train=True, image=True, timestamp=True, audio=True, text=True,
                    weight='place', normalize=True):

        self.df = pd.read_csv(test_path)
        self.labels = list(self.df.scene_id)
        self.labels_set = set(self.df.scene_id.unique())
        self.label_to_indices = {label: np.where(self.df.scene_id == label)[0]
                                for label in self.labels_set}
        # print('self.label_set:', self.labels_set)
        # print('self.labels_to_indices:',  self.label_to_indices)

        if normalize:
            self.start_sec = list(scale(self.df.start_sec))
            self.end_sec   = list(scale(self.df.end_sec))
            self.shot_sec  = list(scale(self.df.shot_sec))
        else:
            self.start_sec = list(self.df.start_sec)
            self.end_sec   = list(self.df.end_sec)
            self.shot_sec  = list(self.df.shot_sec)

        if weight == 'place':
            self.images = list(self.df.place365_feature_path)
        else:
            self.images = list(self.df.imagenet_feature_path)
        self.audios = list(self.df.audio_feature_path)
        self.texts  = list(self.df.text_feature_path)
    
    def get_df(self):
        return self.df
    
    def __getitem__(self, index):
        # print(f'called __getitem__ index:{index}')
        data = {}

        """ load image """
        img = np.load(self.images[index])
        img = torch.from_numpy(img)
        data['image'] = torch.squeeze(img, dim=0)

        """ load audio """
        aud = np.load(self.audios[index])
        aud = torch.from_numpy(aud)
        data['audio'] = torch.squeeze(aud.view(aud.size()[0], -1), dim=0)

        """ load text """
        txt = np.load(self.texts[index])
        data['text'] = torch.from_numpy(txt)

        """ load timestamp """
        timestamp = [self.shot_sec[index], self.start_sec[index], self.end_sec[index]]
        data['timestamp'] = torch.tensor(timestamp)

        return data

    def __len__(self):
        return len(self.df.shot_id)

def predict(dataset, model, cuda, kwards):
    model.eval()
    features = torch.zeros(0)

    df = dataset.get_df()
    print(df)

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **kwards)

    with torch.no_grad():
        for data in data_loader:
            if cuda:
                for d in data:
                    data[d] = data[d].cuda()
            
            feature = torch.Tensor(model.get_embedding(data).data.cpu().numpy())
            features = torch.cat((features, feature))
    
    features = features.view(len(dataset), 128)
    print(features.size())
    features = features.numpy()
    print(features.shape)

    # calculation similarity & distance
    cos_sim_list = []
    euc_dis_list = []
    cos_id_list  = []
    euc_id_list  = []

    cos_scene_id = 0
    euc_scene_id = 0

    for i in range(len(dataset)):
        if i == 0: 
            cos_sim_list.append(0)
            euc_dis_list.append(0)

            cos_id_list.append(cos_scene_id)
            euc_id_list.append(euc_scene_id)
            cos_scene_id +=1
            euc_scene_id +=1
            continue

        sim = cos_sim(features[i-1], features[i])
        dis = eucli_dis(features[i-1], features[i])

        print(f'sim {sim}, {sim.dtype}, {sim.shape}')
        print(f'dis {dis}, {dis.dtype}, {dis.shape}')
        cos_sim_list.append(sim)
        euc_dis_list.append(dis)

        if sim < 0.5: cos_scene_id += 1
        if dis > 0.5: euc_scene_id += 1
        cos_id_list.append(cos_scene_id)
        euc_id_list.append(euc_scene_id)
    
    print(f'cos_sim_list length: {len(cos_sim_list)}')
    print(f'euc_dis_list length: {len(euc_dis_list)}')

    pred_df = pd.DataFrame({
        'shot_id':df.shot_id,
        'scene_id':df.scene_id,
        'cos_sim': cos_sim_list,
        'euc_dis_list':euc_dis_list,
        'cos_scene_id': cos_id_list,
        'euc_scene_id': euc_id_list
    })

    print(pred_df)
    pred_df.to_csv('./pred.csv', index=False)
    return pred_df

def calc_eval(df):
    return

if __name__ == "__main__":
    """ define device """
    cuda = torch.cuda.is_available()
    kwards = {'num_workers':1, 'pin_memory': True} if cuda else {}

    pred_dataset = PredData()

