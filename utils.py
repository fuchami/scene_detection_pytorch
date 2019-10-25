# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset),2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data,target in dataloader:
            print(type(data))
            print(len(data))
            print(len(data[0]))
            print(len(data[1]))
            if cuda:
                data = tuple(d.cuda() for d in data)
            embeddings[k:k+len(data)] = model.get_embedding(data[0], data[1]).data.cpu().numpy()
            labels[k:k+len(data)] = target.numpy()
            k += len(data)
    
    return embeddings, labels