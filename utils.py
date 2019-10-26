# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_embeddings(embeddings, targets, save_path, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(int(max(targets))+1):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, label=i)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.legend()
    plt.savefig(save_path +'embedding.png')

def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset),2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data, _, target in dataloader:
            if cuda:
                data = tuple(d.cuda() for d in data)
            embeddings[k:k+len(data[0])] = model.get_embedding(data[0], data[1]).data.cpu().numpy()
            labels[k:k+len(data[0])] = target.numpy()
            k += len(data[0])
    
    return embeddings, labels