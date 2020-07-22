# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:11:38 2020

@author: thuli
"""


from collections import defaultdict
import torch.nn.functional as F
import time
import copy

#We need define the dice loss function for this specific application
def dice_loss(pred, target, smooth = 1e-7):
    pred = pred.contiguous()
    target = target.contiguous() 
    target=target.float()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

#Define the loss calculation of our training 
def calc_loss(pred,target,metrics,bce_weight=0.5):
    pred=pred.float()
    target=target.float()
    bce=F.binary_cross_entropy_with_logits(pred,target)
    
    pred=F.sigmoid(pred)
    
    dice=dice_loss(pred,target)
    loss=bce*bce_weight+dice * (1-bce_weight)
    metrics['bce'] += bce.data.cpu().numpy() *target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() *target.size(0) 
    return loss
#Define a function which prints the loss we got
def print_metrics(metrics, epoch_samples,phase):
    outputs=[]
    for k in metrics.keys():
        outputs.append("{}:{:4f}".format(k,metrics[k]/epoch_samples))
        print("{}:{}".format(phase,", ".join(outputs)))