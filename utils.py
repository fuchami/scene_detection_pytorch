# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

chars = "^<>vo+d"
markers = [chars[i%7] for i in range(100)]

def plot_embeddings(embeddings, targets, save_path, xlim=None, ylim=None):
    plt.figure(figsize=(20,20))
    for i in range(int(max(targets))+1):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, label=i, marker=markers[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.legend()
    plt.savefig(save_path +'embedding.png')

def tb_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        features = np.zeros((len(dataloader.dataset), 256))
        labels = []
        k = 0
        for data, target, label in dataloader:
            if cuda:
                for dict_ in data:
                    for d in dict_:
                        dict_[d] = dict_[d].cuda()

            features[k:k+len(data[0][i])] = model.get_embedding(*data).data.cpu().numpy()
            labels.append(label)
            k += len(data[0][i])
    
    return torch.Tensor(features), labels

def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset),256))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data, target, label in dataloader:
            if cuda:
                for dict_ in data:
                    for d in dict_:
                        dict_[d] = dict_[d].cuda()

            i = list(data[0].keys())[0]

            embeddings[k:k+len(data[0][i])] = model.get_embedding(*data).data.cpu().numpy()
            labels[k:k+len(data[0][i])] = label.numpy()
            k += len(data[0][i])

    # feature embedding using t-SNE
    print(embeddings.shape)
    print(labels.shape)

    x_reduces = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    print(x_reduces.shape)
    
    return x_reduces, labels