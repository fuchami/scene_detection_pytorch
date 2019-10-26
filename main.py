# coding:utf-8

import numpy as np
import argparse,os
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from trainer import fit
from losses import ContrastiveLoss
from datasets import MultiModalDataset
from models.networks import SiameseNet,ImageExtractor
from utils import extract_embeddings,plot_embeddings


def main(args):
    """ define device """
    cuda = torch.cuda.is_available()
    print('run on cuda?: ', cuda)

    """ setting logs """
    now_time = datetime.now().strftime("%y%m%d_%H%M")
    log_dir_name = f'./logs/{now_time}{args.model}_{args.images}_{args.audio}_{args.text}_'
    log_dir_name += f'epoch{args.epochs}batch{args.batchsize}lr{args.learning_rate}extract_{args.img_model}_'
    log_dir_name += f'margin{args.margin}/'
    print('log_dir_name:', log_dir_name)
    
    if not os.path.exists(log_dir_name): os.makedirs(log_dir_name)
    writer = SummaryWriter(log_dir_name)

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
    # images, labels = next(iter(train_loader))
    # writer.add_graph(model, images)
    if cuda: model.cuda()

    """ define parameters """
    loss_fn = ContrastiveLoss(args.margin)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = args.epochs
    log_interval = args.log_interval

    """ train """
    fit(train_loader, None, model, loss_fn, optimizer,scheduler, n_epochs, cuda, log_interval, writer)
    
    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model, cuda)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline, log_dir_name)

    """ eval """
    torch.save(model.state_dict(), log_dir_name+'weight.pth')

    """ end """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='siamese')
    parser.add_argument('--images',  '-i', default=True)
    parser.add_argument('--audio',  '-a', default=False)
    parser.add_argument('--text',  '-t', default=False)

    parser.add_argument('--epochs', '-e', default=10, type=int)
    parser.add_argument('--batchsize', '-b', default=32, type=int)
    parser.add_argument('--learning_rate', '-r', default=1e-2)
    parser.add_argument('--log_interval', '-l', default=50, type=int)
    parser.add_argument('--img_model', default='res')
    parser.add_argument('--margin', '-m', default=1.)

    args = parser.parse_args()
    main(args)

