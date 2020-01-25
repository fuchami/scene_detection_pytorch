# coding:utf-8

import numpy as np
import argparse,os
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, glob

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from trainer import fit
from eval import PredData, predict, calc_eval, mIoU, add_avg
from losses import ContrastiveLoss,TripletLoss,AngularLoss
from datasets import SiameseMulti,TripletMulti
from models.networks import SiameseNet,TripletNet
from models.embedding import EmbeddingNet
from utils import extract_embeddings,plot_embeddings, tb_embeddings


def main(args):
    """ define device """
    cuda = torch.cuda.is_available()
    print('run on cuda?: ', cuda)

    test_data_name = os.path.splitext(os.path.basename(args.test_path))[0]
    print('test_data_name:', test_data_name)

    """ setting logs """
    now_time = datetime.now().strftime("%y%m")
    base_log_dir_name = f'./logs/{args.model}_{args.merge}_{args.image}-{args.weight}_{args.audio}_{args.text}_{args.time}_'
    base_log_dir_name += f'epoch{args.epochs}batch{args.batchsize}lr{args.learning_rate}_norm_{args.normalize}_{args.optimizer}_outdim{args.outdim}'
    if args.model == 'angular':
        base_log_dir_name += f'_alpha{args.alpha}/'
    else:
        base_log_dir_name += f'_margin{args.margin}/'
    print('base_log_dir_name:', base_log_dir_name)

    log_dir_name = f'{base_log_dir_name}{test_data_name}'
    
    if not os.path.exists(log_dir_name): os.makedirs(log_dir_name)
    writer = SummaryWriter(log_dir_name)

    """ dump hyper parameters """
    # TODO: ハイパラをtxtにdumpする

    """ load dataset """
    if args.model == 'siamese':
        train_dataset = SiameseMulti(train=True, image=args.image, timestamp=args.time, audio=args.audio,
                                    text=args.text, weight=args.weight, normalize=args.normalize,
                                    test_path=args.test_path)
        test_dataset = SiameseMulti(train=False, image=args.image, timestamp=args.time, audio=args.audio,
                                    text=args.text, weight=args.weight, normalize=args.normalize,
                                    test_path=args.test_path)
    else: 
        train_dataset = TripletMulti(train=True, image=args.image, timestamp=args.time, audio=args.audio,
                                    text=args.text, weight=args.weight, normalize=args.normalize,
                                    test_path=args.test_path)
        test_dataset = TripletMulti(train=False, image=args.image, timestamp=args.time, audio=args.audio,
                                    text=args.text, weight=args.weight, normalize=args.normalize,
                                    test_path=args.test_path)

    kwards = {'num_workers':1, 'pin_memory': True} if cuda else {}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            drop_last=True,
                                            **kwards)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            **kwards)

    tb_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **kwards)

    print('train_dataset length', len(train_dataset))
    print('test_dataset length', len(test_dataset))

    pred_dataset = PredData(train=False, image=args.image, timestamp=args.time, audio=args.audio,
                                text=args.text, weight=args.weight, normalize=args.normalize,
                                test_path=args.test_path)

    """ build model """
    if args.model == 'siamese':
        model = SiameseNet(image=args.image, audio=args.audio, text=args.text, time=args.time,
                            merge=args.merge, outdim=args.outdim)
        loss_fn = ContrastiveLoss(args.margin)
    elif args.model == 'triplet':
        model = TripletNet(image=args.image, audio=args.audio, text=args.text, time=args.time,
                            merge=args.merge, outdim=args.outdim)
        loss_fn = TripletLoss(args.margin)
    elif args.model == 'angular':
        model = TripletNet(image=args.image, audio=args.audio, text=args.text, time=args.time,
                            merge=args.merge, outdim=args.outdim)
        loss_fn = AngularLoss()

    """ tensorboad add graph """
    data, _, labels, _ = next(iter(train_loader))
    writer.add_graph(model, data)
    if cuda: model.cuda()

    """ define parameters """
    lr = args.learning_rate
    if args.optimizer == 'adam':
        print(f'=== optimizer adam ===')
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'sgd':
        print(f'=== optimizer sgd ===')
        # referenceに乗っ取る
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    n_epochs = args.epochs
    log_interval = args.log_interval

    """ train """
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, writer)
    
    """ tsne embedding plot """
    # test_embeddings_baseline, test_labels_baseline = extract_embeddings(test_loader, model, cuda, args.outdim)
    # plot_embeddings(test_embeddings_baseline, test_labels_baseline, log_dir_name)

    """ tensorboard embedding """
    # features, labels, label_imgs= tb_embeddings(tb_loader, test_dataset, model, cuda, args.outdim)
    # writer.add_embedding(features, metadata=labels, label_img=label_imgs)

    print('======= eval =======')
    pred_df = predict(args.outdim, pred_dataset, model, cuda, kwards)
    pred_df.to_csv(log_dir_name+'pred.csv', index=False)
    miou = mIoU(pred_df)
    print('mIoU:', miou)

    with open(f'{base_log_dir_name}/mIoU.csv', 'a') as f:
        print(f'{log_dir_name}, {miou}', file=f)

    torch.save(model.state_dict(), log_dir_name+'weight.pth')
    writer.close()

    print('=================== finished ===================')

if __name__ == "__main__":
    # All test data cross validation
    test_path_list = sorted(glob.glob('./BBC_Planet_Earth_Dataset/dataset/annotator_0/*'))
    for test_path in test_path_list:
        print('****************************************************************************************************')
        parser = argparse.ArgumentParser(description='train SiameseNet or TripletNet')
        parser.add_argument('--model', default='triplet',
                            help='siamese or triplet or angular')
        parser.add_argument('--image', action='store_true')
        parser.add_argument('--audio', action='store_true')
        parser.add_argument('--text', action='store_true')
        parser.add_argument('--time', action='store_true')
        parser.add_argument('--normalize', action='store_false')
        parser.add_argument('--outdim', default=32, type=int)

        parser.add_argument('--margin', default=0.1, type=float)
        parser.add_argument('--alpha', type=int, default=36, help='angular loss alpha 36 or 45')
        parser.add_argument('--merge', default='concat', type=str, help='chose vector merge concat or mcb')

        parser.add_argument('--weight', default='place', type=str, help='chose place or imagenet trained weight')

        parser.add_argument('--epochs', '-e', default=300, type=int)
        parser.add_argument('--batchsize', '-b', default=128, type=int)
        parser.add_argument('--learning_rate', '-r', default=0.01)
        parser.add_argument('--log_interval', '-l', default=100, type=int)
        parser.add_argument('--optimizer', '-o' ,default='sgd')

        parser.add_argument('--test_path', default=test_path)

        args = parser.parse_args()
        main(args)

    # 最後に平均のmIoU値を出す
    # now_time = datetime.now().strftime("%y%m")
    dir_name = f'./logs/{args.model}_{args.merge}_{args.image}-{args.weight}_{args.audio}_{args.text}_{args.time}_'
    dir_name += f'epoch{args.epochs}batch{args.batchsize}lr{args.learning_rate}_norm_{args.normalize}_{args.optimizer}_outdim{args.outdim}'
    if args.model == 'angular':
        dir_name += f'_alpha{args.alpha}/'
    else:
        dir_name += f'_margin{args.margin}/'
    
    add_avg(dir_name+'mIoU.csv')
    print('end...****************************************************************************************************')
