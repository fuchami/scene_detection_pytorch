# coding:utf-8

import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

from trainer import fit
from losses import ContrastiveLoss
from datasets import MultiModalDataset
from models.networks import SiameseNet,ImageExtractor

cuda = torch.cuda.is_available()

def main(args):
    """ setting losg """

    """ load dataset """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    normalize])
    
    train_dataset = MultiModalDataset(csv_path=None, transform=transform, train=True, audio=False)
    kwards = {'num_workers':1, 'pin_memory': True} if cuda else {}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            **kwards)

    """ build model """
    image_extractor = ImageExtractor(model=args.img_model)
    model = SiameseNet(image_extractor)
    if cuda: model.cuda()

    """ define parameters """
    loss_fn = ContrastiveLoss(args.margin)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = args.epochs
    log_interval = args.log_interval

    """ train """
    fit(train_loader, None, model, loss_fn, optimizer,scheduler, 
            n_epochs, cuda, log_interval)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images',  '-i', default=True)
    parser.add_argument('--audio',  '-a', default=False)
    parser.add_argument('--text',  '-t', default=False)

    parser.add_argument('--epochs', '-e', default=10)
    parser.add_argument('--batchsize', '-b', default=64)
    parser.add_argument('--learning_rate', '-r', default=1e-2)
    parser.add_argument('--log_interval', '-l', default=50)
    parser.add_argument('--img_model', default='res')
    parser.add_argument('--margin', '-m', default=1.)

    args = parser.parse_args()
    main(args)

