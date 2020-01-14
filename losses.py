# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 
    if samples are from the same class and label == 0 otherwise
    サンプルが同じクラスからのものである場合は2つのサンプルとターゲットラベル== 1の埋め込みを受け取り、
    そうでない場合はラベル== 0を受け取ります
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    
    def forward(self, output1, output2, target, size_average=True, norm=True):
        distances = (output2 - output1).pow(2).sum(1) # squared distance
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * 
                        F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet Loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    アンカーサンプル、ポジティブサンプル、ネガティブサンプルの埋め込みを取得します
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative, size_average=True, norm=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        loss_embedd = anchor.norm(2) + positive.norm(2) + negative.norm(2)

        return losses.mean() + 0.001 * loss_embedd if norm else losses.mean()
    
class AngularLoss(nn.Module):
    # reference: https://qiita.com/tomp/items/0f1762e5971f4768922d
    def __init__(self, alpha=45, in_degree=True):
        super(AngularLoss, self).__init__()
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.tan_alpha = np.tan(alpha) ** 2
    
    def forward(self, anchor, positive, negative, norm=True):
        c = (anchor + positive) / 2
        sq_dist_ap = (anchor - positive).pow(2).sum(1)
        sq_dist_nc = (negative - c).pow(2).sum(1)
        loss = sq_dist_ap - 4*self.tan_alpha*sq_dist_nc
        loss_embedd = anchor.norm(2) + positive.norm(2) + negative.norm(2)

        return F.relu(loss).mean() + 0.001 * loss_embedd if norm else F.relu(loss).mean()
