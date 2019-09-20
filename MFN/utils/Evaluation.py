#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Evaluation of LFW data

@author: RockerAI
"""
import numpy as np
import torch

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def getAccuracy(scores, flags, threshold, method):
    if method == 'l2_distance':
        p = np.sum(scores[flags == 1] < threshold)
        n = np.sum(scores[flags == -1] > threshold)
    elif method == 'cos_distance':
        p = np.sum(scores[flags == 1] > threshold)
        n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum, method):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i], method)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def getFeature(net, dataloader, device, flip = True):
### Calculate the features ###
    featureLs = None
    featureRs = None 
    count = 0
    for det in dataloader:
        for i in range(len(det)):
            det[i] = det[i].to(device)
        count += det[0].size(0)
#        print('extracing deep features from the face pair {}...'.format(count))
    
        with torch.no_grad():
            res = [net(d).data.cpu() for d in det]
            
        if flip:      
            featureL = l2_norm(res[0] + res[1])
            featureR = l2_norm(res[2] + res[3])
        else:
            featureL = res[0]
            featureR = res[2]
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = torch.cat((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = torch.cat((featureRs, featureR), 0)
        
    return featureLs, featureRs

def evaluation_10_fold(featureL, featureR, dataset, method = 'l2_distance'):
    
    ### Evaluate the accuracy ###
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = np.array(dataset.folds).reshape(1,-1)
    flags = np.array(dataset.flags).reshape(1,-1)
    featureLs = featureL.numpy()
    featureRs = featureR.numpy()

    for i in range(10):
        
        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)
        
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == 'l2_distance':
            scores = np.sum(np.power((featureLs - featureRs), 2), 1) # L2 distance
        elif method == 'cos_distance':
            scores = np.sum(np.multiply(featureLs, featureRs), 1) # cos distance
        
        threshold[i] = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000, method)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold[i], method)
        
    return ACCs, threshold
