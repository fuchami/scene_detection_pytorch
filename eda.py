# coding:utf-8

#%%
import os,sys,glob,csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%% シーン分割点のアノテーション
scene_label_path = "./BBC_Planet_Earth_Dataset/annotations/scenes/"
annotator_list = ['annotator_'+str(i) for i in range(5)]
movie_name_list = [ os.path.basename(i) for i in sorted(glob.glob(scene_label_path+annotator_list[0]+"/*")) ]

print(movie_name_list)

#%%
def load_scene_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    data = data[0].split(',')
    return data

#%% 
def plot_movie_scene_split(movie_name):
    data = {}
    one_hot_data = {}

    for annotator in annotator_list:
        label_path = f'{scene_label_path}{annotator}/{movie_name}'
        data[annotator] = load_scene_txt(label_path)

        scene_length = int(data[annotator][-1])
        one_hot_data[annotator] = []
        for i in range(scene_length + 1):
            if str(i) in data[annotator]:
                one_hot_data[annotator].append(1)
            else:
                one_hot_data[annotator].append(0)
        
        # print(data[annotator])
        # print(one_hot_data[annotator])
        # one_hot_data[annotator] = np.array(one_hot_data[annotator])

    #%% convert 2d np.array 
    orderNames = list(one_hot_data.keys())
    dataMatrix = np.array([list(one_hot_data[i]) for i in orderNames])
    # print(dataMatrix[0].shape)
    print(dataMatrix.shape[1])

    #%%
    plt.style.use('default')
    plt.figure(figsize=(15,5))
    plt.title(f'{movie_name} SceneSplit Annotation')
    sns.set()
    sns.set_style('whitegrid')

    df = pd.DataFrame(dataMatrix,
                        index=annotator_list,
                        columns=[str(i) for i in range(dataMatrix.shape[1])])

    sns.heatmap(df, cbar=False, cmap='plasma')
    # plt.show()
    movie_name,_ = os.path.splitext(os.path.basename(movie_name))
    plt.savefig(f'./image/eda/{movie_name}.png')
#%%
for movie_name in movie_name_list:
    plot_movie_scene_split(movie_name)

