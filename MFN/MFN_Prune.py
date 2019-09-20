#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:23:05 2019

@author: xingyu
"""
import sys
sys.path.append('..')
import time
import numpy as np
import argparse
import math 
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.utils.data as data
from Base_Model.face_model import *
from utils.FilterPrunner import FilterPrunner
from utils.prune_MFN import prune_MFN
from data_set.dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
from utils.Evaluation import getFeature, evaluation_10_fold

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def load_data(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    root = '../data_set/LFW/lfw_align_112'
    file_list = '../data_set/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '../data_set/CFP-FP/CFP_FP_aligned_112'
    file_list = '../data_set/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '../data_set/AgeDB-30/agedb30_align_112'
    file_list = '../data_set/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = '../data_set/CASIA_Webface_Image'
        file_list = '../data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = '../data_set/faces_emore_images'
        file_list = '../data_set/faces_emore_images/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=transform) 
    
    else:
        raise NameError('no training data exist!')
    
    dataloaders = {'train': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size, shuffle=False, num_workers=2),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size, shuffle=False, num_workers=2),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size, shuffle=False, num_workers=2)}
    
    dataset = {'train': dataset_train,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train), 'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

def evalu(model):
    
    model.eval()
    for phase in ['LFW', 'CFP_FP', 'AgeDB30']:                 
        featureLs, featureRs = getFeature(model, dataloaders[phase], device, flip = True)
        ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = 'l2_distance')
        print('{} average acc:{:.4f} average threshold:{:.4f}'
              .format(phase, np.mean(ACCs) * 100, np.mean(threshold)))

def train(model, epoches=2):
    
    margin = Arcface(embedding_size=512, classnum=int(dataset['train'].class_nums),  s=32., m=0.5).to(device)
    checkpoint = torch.load("Base_Model/Iter_528000_margin.ckpt", map_location=lambda storage, loc: storage)
    margin.load_state_dict(checkpoint['net_state_dict'])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.001, momentum=0.9, nesterov=True)
    
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[1], gamma=0.3)
    
    total_iters = 0
    for epoch in range(epoches):
        # train model
        exp_lr_scheduler.step()
        model.train()     
        for det in dataloaders['train']: 
            img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()
            
            with torch.set_grad_enabled(True):
                raw_logits = model(img)
                output = margin(raw_logits, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer_ft.step()
                
                total_iters += 1
                # print train information
                if total_iters % 1000 == 0:
                    # current training accuracy 
                    _, preds = torch.max(output.data, 1)
                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  

                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}"
                          .format(epoch, epoches-1, total_iters, loss.item(), correct/total))

def prune_model(model, prunner):

    model.train() 
    margin = Arcface(embedding_size=512, classnum=int(dataset['train'].class_nums),  s=32., m=0.5).to(device)
    checkpoint = torch.load("Base_Model/Iter_528000_margin.ckpt", map_location=lambda storage, loc: storage)
    margin.load_state_dict(checkpoint['net_state_dict'])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    prunner.reset()
        
    for i_batch, det in enumerate(dataloaders['train']):
            
        printProgressBar(i_batch, 10000, prefix = 'Progress:', suffix = 'Complete', length = 50)
        img, label = det[0].to(device), det[1].to(device)

        # zero the parameter gradients
        model.zero_grad()
    
        with torch.set_grad_enabled(True):
            raw_logits = prunner.forward(img)
            output = margin(raw_logits, label)
            loss = criterion(output, label)
            loss.backward()
        if i_batch == 10000: # only use 1/10 train data
            break
                   
    prunner.normalize_ranks_per_layer()
    filters_to_prune = prunner.get_prunning_plan(args.filter_size)
    
    return filters_to_prune  

def rearrange(model):
    
    index = 0
    modules = {}
    for names, module in list(model._modules.items()):
        if isinstance(module, Depth_Wise):
            for _, module_sub in list(module._modules.items()):
                modules[index] = module_sub
                index += 1
                
        elif isinstance(module, Residual):
            for i in range(len(module.model)):
                for _, model_sub_sub in list(module.model[i]._modules.items()):
                    modules[index] = model_sub_sub
                    index += 1          
        else:
            modules[index] = module
            index += 1
    
    return modules
    
def total_num_filters(modules):
    
    filters = 0
    for name, module in modules.items():
        if isinstance(module, Conv_block) or isinstance(module, Linear_block):
            filters = filters + module.conv.out_channels
            
    return filters
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 64, help='batch size for training and evaluation')
    parser.add_argument("--filter_size", type = int, default = 512)
    parser.add_argument("--filter_percentage", type = float, default = 0.5)  
    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloaders , dataset_sizes, dataset = load_data(args.batch_size) 
    model = MobileFaceNet(512).to(device)
    model.load_state_dict(torch.load('Base_Model/MobileFace_Net', map_location=lambda storage, loc: storage))
    
    modules = rearrange(model)
      
    save_dir = 'saving_MFN_prunning_result'
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    
    print("Check the initial model accuracy...")
    since = time.time()
    evalu(model)
    print("initial test :: time cost is {:.2f} s".format(time.time()-since))
    
    #Make sure all the layers are trainable
    for param in model.parameters():
        param.requires_grad = True
        
    number_of_filters = total_num_filters(modules) 
    print("total model conv2D filters are: ", number_of_filters)
    
    num_filters_to_prune_per_iteration = args.filter_size
    
    iterations = math.ceil((float(number_of_filters) * args.filter_percentage) / (num_filters_to_prune_per_iteration+1e-6))
    print("Number of iterations to prune {} % filters:".format(args.filter_percentage*100), iterations)
    
    for it in range(iterations):
        
        print("iter{}. Ranking filters ..".format(it))
        
        prunner = FilterPrunner(modules, use_cuda = True) 
        filters_to_prune = prune_model(model, prunner)
        
        layers_prunned = [(k, len(filters_to_prune[k])) for k in sorted(filters_to_prune.keys())] # k: layer index, number of filters
        print("iter{}. Layers that will be prunned".format(it), layers_prunned)
        
        print("iter{}. Prunning filters.. ".format(it))
        for layer_index, filter_index in filters_to_prune.items():
            model, modules = prune_MFN(model, layer_index, *filter_index, use_cuda=True)
            
        model = model.to(device)
            
        print("iter{}. {:.2f}% Filters remaining".format(it, 100*float(total_num_filters(modules)) / number_of_filters))     
        
        print("iter{}. without retrain...".format(it))
        evalu(model)
            
        print("iter{}. Fine tuning to recover from prunning iteration.. ".format(it))
        torch.cuda.empty_cache()
        train(model, epoches = 2)
        
        print("iter{}. after retrain...".format(it))
        since = time.time()
        evalu(model)
        print("iter{}. test time cost is {:.2f} s".format(it, time.time()-since))
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'MFN_weights_pruned_{}'.format(it)))
        torch.save(model, os.path.join(save_dir, 'MFN_prunned_{}'.format(it)))
        
    print("Finished prunning")