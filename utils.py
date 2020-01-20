# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
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

def tb_embeddings(dataloader,dataset, model, cuda, outdim):
    print('=== tensorboard embedding ===')

    tb_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((50, 50)),
        torchvision.transforms.ToTensor(),
    ])

    # features = torch.zeros(0)
    features = torch.zeros(0)
    label_imgs = torch.zeros(0)
    labels = []

    with torch.no_grad():
        model.eval()
        k = 0
        for data, target, label, img1_path in dataloader:
            if cuda:
                for dict_ in data:
                    for d in dict_:
                        dict_[d] = dict_[d].cuda()
            
            feature = torch.Tensor(model.get_embedding(data[0]).data.cpu().numpy())
            features = torch.cat((features, feature))
            labels.append(label)
            label_img = tb_transform(PIL.Image.open(img1_path[0]).convert('RGB'))
            label_imgs = torch.cat((label_imgs, label_img))
    
    features = features.view(len(dataset), outdim)
    label_imgs = label_imgs.view(len(dataset), 3, 50, 50)
    
    return features, labels, label_imgs

def extract_embeddings(dataloader, model, cuda, outdim):
    print('== extract_embeddings ==')
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), outdim))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data, target, label, _ in dataloader:
            if cuda:
                for dict_ in data:
                    for d in dict_:
                        dict_[d] = dict_[d].cuda()

            i = list(data[0].keys())[0]

            embeddings[k:k+len(data[0][i])] = model.get_embedding(data[0]).data.cpu().numpy()
            labels[k:k+len(data[0][i])] = label.numpy()
            # print('labels', labels)
            k += len(data[0][i])

    # feature embedding using t-SNE
    # print(embeddings.shape)
    # print(labels.shape)

    x_reduces = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    # print(x_reduces.shape)
    
    return x_reduces, labels