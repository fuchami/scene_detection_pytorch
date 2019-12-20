# coding:utf-8
"""
image: resnet-50
audio: VGGish
text: BERT
"""

#%% 
import os, sys, glob, csv
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from models.extractor import ImageExtractor
from torchvggish import vggish_input, vggish
from transformers import BertModel, BertTokenizer

#%% extra feature
class ExtraFeature(object):
    def __init__(self, annotator, movie_name):
        """ define device """
        cuda = torch.cuda.is_available()
        print('run on cuda?', cuda)

        """ load csv """
        self.episode = movie_name.split('_', 1)[0] # 01~11の値
        self.csv_path = pd.read_csv(f'./BBC_Planet_Earth_Dataset/dataset/annotator_{annotator}/{movie_name}.csv')
        print(self.csv_path.head())


        image_feature_dir = f'./BBC_Planet_Earth_Dataset/feature/image/'
        audio_feature_dir = f'./BBC_Planet_Earth_Dataset/feature/audio/'
        text_feature_dir = f'./BBC_Planet_Earth_Dataset/feature/text/'
        if not os.path.exists(image_feature_dir): os.makedirs(image_feature_dir)
        if not os.path.exists(audio_feature_dir): os.makedirs(audio_feature_dir)
        if not os.path.exists(text_feature_dir): os.makedirs(text_feature_dir)

        self.image_feature_path = []
        self.audio_feature_path = []
        self.text_feature_path =  []

        """ image extra """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize])
        self.image_extractor = ImageExtractor('res')
        self.image_extractor.eval()
        if cuda: self.image_extractor.cuda()

        self.image_extra()
        if len(self.image_feature_path) > 1:
            self.csv_path['image_feature_path'] = self.image_feature_path

        """ audio extra """
        self.audio_extractor = vggish()
        self.audio_extractor.eval()
        if cuda: self.audio_extractor

        self.audio_extra()
        if len(self.audio_feature_path) > 0:
            self.csv_path['audio_feature_path'] = self.audio_feature_path

        """ text extra """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        if cuda: self.bert.cuda()

        self.text_extra()
        if len(self.text_feature_path) > 0:
            self.csv_path['text_feature_path'] = self.text_feature_path

        """ finished """
        print(self.csv_path.head())
        self.csv_path.to_csv(f'./BBC_Planet_Earth_Dataset/dataset/annotator_{annotator}/{movie_name}.csv', index=None)
    
    def text_extra(self):
        for row in self.csv_path.itertuples():
            print(row.text)
            print(row.shot_id)
            txt_feature_path = f'./BBC_Planet_Earth_Dataset/feature/text/bbc_{self.episode}/{row.shot_id}'
            txt_feature_dir = os.path.dirname(txt_feature_path)
            if not os.path.exists(txt_feature_dir): os.makedirs(txt_feature_dir)
            print(txt_feature_path)

            # ファイルがなければ特徴抽出
            if os.path.isfile(txt_feature_path+'.npy'):
                print('already exist text feature as npy: ', txt_feature_path)
            else:
                txt = torch.tensor([self.tokenizer.encode(row.text, add_special_tokens=True)]).cuda()
                with torch.no_grad():
                    txt_emb = self.bert(txt)[0][0]

                txt_emb = torch.mean(txt_emb, dim=0)
                print('text extract size:', txt_emb.size())
                txt_emb = txt_emb.cpu().detach().numpy()
                np.save(txt_feature_path,  txt_emb)
                print('save text features as npy :', txt_feature_path)
            self.text_feature_path.append(txt_feature_path+'.npy')
    
    def image_extra(self):
        for row in self.csv_path.itertuples():
            print(row.image)
            img_feature_path = row.image.replace('.jpg', '').replace('frame', 'feature/image')
            img_feature_dir = os.path.dirname(img_feature_path)
            if not os.path.exists(img_feature_dir): os.makedirs(img_feature_dir)
            print(img_feature_path)

            # ファイルがなければ特徴抽出
            if os.path.isfile(img_feature_path+'.npy'):
                print('already exist image feature as npy: ', img_feature_path)
            else: 
                img_feature = self.image_extractor(self.image_open(row.image))
                img_feature = img_feature.cpu().detach().numpy()
                np.save(img_feature_path, img_feature)
                print('save image feature as npy:', img_feature)
            self.image_feature_path.append(img_feature_path+'.npy')
    
    def audio_extra(self):
        for row in self.csv_path.itertuples():
            print(row.audio)
            aud_feature_path = row.audio.replace('.wav', '').replace('audio', 'feature/audio')
            aud_feature_dir = os.path.dirname(aud_feature_path)
            if not os.path.exists(aud_feature_dir): os.makedirs(aud_feature_dir)
            print(aud_feature_path)

            if os.path.isfile(aud_feature_path+'.npy'):
                print('already exist audio feature as npy: ', aud_feature_path)
            else: 
                aud = vggish_input.wavfile_to_examples(row.audio)
                with torch.no_grad():
                    aud_feature = self.audio_extractor.forward(aud)

                aud_feature = aud_feature.view(1, 20, -1)
                print('audio extract size: ', aud_feature.size()) # [bs, ]
                aud_feature = aud_feature.cpu().detach().numpy()
                np.save(aud_feature_path, aud_feature)
                print('save audio feature as npz:', aud_feature)
            self.audio_feature_path.append(aud_feature_path+'.npy')
        
    def image_open(self, t):
        image = Image.open(t)
        image = self.transform(image)
        image = torch.unsqueeze(image, 0).cuda()
        # print(image.size())
        return image

#%%
# extrafeature = ExtraFeature('0', '01_From_Pole_to_Pole')

# %%
movie_name_list = [os.path.basename(i) for i in sorted(glob.glob('./BBC_Planet_Earth_Dataset/annotations/shots/*'))]
annotators_list = [str(i)  for i in range(5)]

for annotator in annotators_list:
    for movie_name in movie_name_list:
        movie_name,_ = os.path.splitext(movie_name)
        extrafeature = ExtraFeature(annotator, movie_name)

#%%