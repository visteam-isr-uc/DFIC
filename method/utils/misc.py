#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:51:53 2023

@author: visteamer
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d

def get_eer(labels_ini, predictions_ini, supressions_ini):
    
    AUCs = []
    EERs = []
    HTERs = []

    for col in range(labels_ini.shape[1]):

        labels = labels_ini[:,col]
        labels = labels[supressions_ini[:, col] == 1]

        predictions = predictions_ini[:,col]
        predictions = predictions[supressions_ini[:, col] == 1]
        
        fpr, tpr, ths = roc_curve(labels, predictions, drop_intermediate = False)
        


        interp_func = interp1d(fpr, tpr)

        fpr_interp = np.arange(0, 1, 0.001)
        tpr_interp = interp_func(fpr_interp)

        znew = abs(fpr_interp + tpr_interp -1)
        eer = 1 - tpr_interp[np.argmin(znew)]
        EERs.append(float(eer))

        best_hter = np.min((fpr + 1 - tpr)/2)
        HTERs.append(float(best_hter))

        auc = roc_auc_score(labels, predictions)
        AUCs.append(float(auc))
        
 
        
    return AUCs, EERs, HTERs
            

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


def adjust_learning_rate(optimizer, epoch, every_decay):
    lr = optimizer.param_groups[0]['lr']
    min_lr=0.00001
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch % every_decay)==0 and epoch>0 and lr>min_lr:
        lr = lr / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    return optimizer


def switch_binary_code(input_list):
    return [np.abs(x-1) for x in input_list]

def deNormalize(tensor_im, means, stds):
    r=(tensor_im[0,:,:]*stds[0])+means[0]
    g=(tensor_im[1,:,:]*stds[1])+means[1]
    b=(tensor_im[2,:,:]*stds[2])+means[2]
    
    new_tensor=tensor_im.clone()
    new_tensor[0,:,:]=r
    new_tensor[1,:,:]=g
    new_tensor[2,:,:]=b
    
    return new_tensor

