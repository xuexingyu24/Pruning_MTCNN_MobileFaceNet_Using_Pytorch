#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:19:30 2019

@author: xingyu
"""
import math
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Base_Model.MTCNN_nets import PNet
from utils.FilterPrunner import FilterPrunner
from utils.prune_mtcnn import prune_mtcnn
from utils.util import ListDataset, printProgressBar, total_num_filters

def test(model, path):
       
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(ListDataset(path), batch_size=batch_size, shuffle=True)
    dataset_sizes = len(ListDataset(path))
    
    model.eval()    
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

    running_correct = 0
    running_gt = 0
    running_loss, running_loss_cls, running_loss_offset = 0.0, 0.0, 0.0
    
    for i_batch, sample_batched in enumerate(dataloader):
        
        printProgressBar(i_batch + 1, dataset_sizes // batch_size + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        input_images, gt_label, gt_offset = sample_batched['input_img'], sample_batched['label'], sample_batched['bbox_target']
        input_images = input_images.to(device)
        gt_label = gt_label.to(device)
        gt_offset = gt_offset.type(torch.FloatTensor).to(device)
        
        with torch.set_grad_enabled(False):
            pred_offsets, pred_label = model(input_images)
            pred_offsets = torch.squeeze(pred_offsets)
            pred_label = torch.squeeze(pred_label)
        
            mask_cls = torch.ge(gt_label, 0)
            valid_gt_label = gt_label[mask_cls]
            valid_pred_label = pred_label[mask_cls]
            
            unmask = torch.eq(gt_label, 0)
            mask_offset = torch.eq(unmask, 0)
            valid_gt_offset = gt_offset[mask_offset]
            valid_pred_offset = pred_offsets[mask_offset]
            
            loss = torch.tensor(0.0).to(device)
            num_gt = len(valid_gt_label)

            if len(valid_gt_label) != 0:
                loss += 0.02*loss_cls(valid_pred_label, valid_gt_label)
                cls_loss = loss_cls(valid_pred_label, valid_gt_label).item()
                pred = torch.max(valid_pred_label, 1)[1]
                eval_correct = (pred == valid_gt_label).sum().item()

            if len(valid_gt_offset) != 0:
                loss += 0.6*loss_offset(valid_pred_offset, valid_gt_offset)
                offset_loss = loss_offset(valid_pred_offset, valid_gt_offset).item()
                
            # statistics
            running_loss += loss.item()*batch_size
            running_loss_cls += cls_loss*batch_size
            running_loss_offset += offset_loss*batch_size
            running_correct += eval_correct
            running_gt += num_gt

    epoch_loss = running_loss / dataset_sizes
    epoch_loss_cls = running_loss_cls / dataset_sizes
    epoch_loss_offset = running_loss_offset / dataset_sizes
    epoch_accuracy = running_correct / (running_gt + 1e-16)
    
    return epoch_accuracy, epoch_loss, epoch_loss_cls, epoch_loss_offset

def train(model, path, epoch=10):
    
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(ListDataset(path), batch_size=batch_size, shuffle=True)
    dataset_sizes = len(ListDataset(path))

    model.train()
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        
        running_loss, running_loss_cls, running_loss_offset = 0.0, 0.0, 0.0
        running_correct = 0.0
        running_gt = 0.0
        
        for i_batch, sample_batched in enumerate(dataloader):
            
            printProgressBar(i_batch + 1, dataset_sizes // batch_size + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

            input_images, gt_label, gt_offset = sample_batched['input_img'], sample_batched[
                'label'], sample_batched['bbox_target']
            input_images = input_images.to(device)
            gt_label = gt_label.to(device)
            gt_offset = gt_offset.type(torch.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                pred_offsets, pred_label = model(input_images)
                pred_offsets = torch.squeeze(pred_offsets)
                pred_label = torch.squeeze(pred_label)
                # calculate the cls loss
                # get the mask element which >= 0, only 0 and 1 can effect the detection loss
                mask_cls = torch.ge(gt_label, 0)
                valid_gt_label = gt_label[mask_cls]
                valid_pred_label = pred_label[mask_cls]

                # calculate the box loss
                # get the mask element which != 0
                unmask = torch.eq(gt_label, 0)
                mask_offset = torch.eq(unmask, 0)
                valid_gt_offset = gt_offset[mask_offset]
                valid_pred_offset = pred_offsets[mask_offset]

                loss = torch.tensor(0.0).to(device)
                cls_loss, offset_loss = 0.0, 0.0
                eval_correct = 0.0
                num_gt = len(valid_gt_label)

                if len(valid_gt_label) != 0:
                    loss += 0.02*loss_cls(valid_pred_label, valid_gt_label)
                    cls_loss = loss_cls(valid_pred_label, valid_gt_label).item()
                    pred = torch.max(valid_pred_label, 1)[1]
                    eval_correct = (pred == valid_gt_label).sum().item()

                if len(valid_gt_offset) != 0:
                    loss += 0.6*loss_offset(valid_pred_offset, valid_gt_offset)
                    offset_loss = loss_offset(valid_pred_offset, valid_gt_offset).item()

                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item()*batch_size
                running_loss_cls += cls_loss*batch_size
                running_loss_offset += offset_loss*batch_size
                running_correct += eval_correct
                running_gt += num_gt

        epoch_loss = running_loss / dataset_sizes
        epoch_loss_cls = running_loss_cls / dataset_sizes
        epoch_loss_offset = running_loss_offset / dataset_sizes
        epoch_accuracy = running_correct / (running_gt + 1e-16)

        print('accuracy: {:.4f} loss: {:.4f} cls Loss: {:.4f} offset Loss: {:.4f}'
              .format(epoch_accuracy, epoch_loss, epoch_loss_cls, epoch_loss_offset))

def prune_model(model, prunner, path):
    
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(ListDataset(path), batch_size=batch_size, shuffle=True)
    dataset_sizes = len(ListDataset(path))
    
    model.train() 
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()
    
    prunner.reset()
        
    for i_batch, sample_batched in enumerate(dataloader):
            
        printProgressBar(i_batch + 1, dataset_sizes // batch_size + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

        input_images, gt_label, gt_offset = sample_batched['input_img'], sample_batched['label'], sample_batched['bbox_target']
        input_images = input_images.to(device)
        gt_label = gt_label.to(device)
        gt_offset = gt_offset.type(torch.FloatTensor).to(device)
    
        # zero the parameter gradients
        model.zero_grad()
    
        with torch.set_grad_enabled(True):
            _, pred_offsets, pred_label = prunner.forward(input_images)
            pred_offsets = torch.squeeze(pred_offsets)
            pred_label = torch.squeeze(pred_label)
            # calculate the cls loss
            # get the mask element which >= 0, only 0 and 1 can effect the detection loss
            mask_cls = torch.ge(gt_label, 0)
            valid_gt_label = gt_label[mask_cls]
            valid_pred_label = pred_label[mask_cls]
    
            # calculate the box loss
            # get the mask element which != 0
            unmask = torch.eq(gt_label, 0)
            mask_offset = torch.eq(unmask, 0)
            valid_gt_offset = gt_offset[mask_offset]
            valid_pred_offset = pred_offsets[mask_offset]
    
            loss = torch.tensor(0.0).to(device)
    
            if len(valid_gt_label) != 0:
                loss += 0.02*loss_cls(valid_pred_label, valid_gt_label)
    
            if len(valid_gt_offset) != 0:
                loss += 0.6*loss_offset(valid_pred_offset, valid_gt_offset)
    
            loss.backward()
        
    prunner.normalize_ranks_per_layer()
    filters_to_prune = prunner.get_prunning_plan(args.filter_size)
    
    return filters_to_prune
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type = str, default = "../data_preprocessing/anno_store/imglist_anno_12.txt")
    parser.add_argument("--test_path", type = str, default = "../data_preprocessing/anno_store/imglist_anno_12_val.txt")
    parser.add_argument("--filter_size", type = int, default = 10)
    parser.add_argument("--filter_percentage", type = float, default = 0.5)  
    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = PNet(is_train=True).to(device)
    model.load_state_dict(torch.load("Base_Model/pnet_Weights", map_location=lambda storage, loc: storage))
    
    prunner = FilterPrunner(model, use_cuda = True) 
    
    save_dir = 'saving_pnet_prunning_result'
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    
    print("Check the initial model accuracy")
    since = time.time()
    accuracy, loss, loss_cls, loss_offset = test(model, args.test_path)
    print('initial test :: accuracy: {:.4f} loss: {:.4f} cls loss: {:.4f} offset loss: {:.4f}'.format(accuracy, loss, loss_cls, loss_offset))
    print("initial test :: time cost is {:.2f} s".format(time.time()-since))
    
    #Make sure all the layers are trainable
    for param in model.features.parameters():
        param.requires_grad = True
        
    number_of_filters = total_num_filters(model) 
    print("total model conv2D filters are: ", number_of_filters)
    
    num_filters_to_prune_per_iteration = args.filter_size
    
    iterations = math.ceil((float(number_of_filters) * args.filter_percentage) / num_filters_to_prune_per_iteration)
    print("Number of iterations to prune {} % filters:".format(args.filter_percentage*100), iterations)
    
    for it in range(iterations):
        
        print("iter{}. Ranking filters ..".format(it))
        filters_to_prune = prune_model(model, prunner, args.test_path)
        
        layers_prunned = [(k, len(filters_to_prune[k])) for k in sorted(filters_to_prune.keys())] # k: layer index, number of filters
        print("iter{}. Layers that will be prunned".format(it), layers_prunned)
        
        print("iter{}. Prunning filters.. ".format(it))
        for layer_index, filter_index in filters_to_prune.items():
            model = prune_mtcnn(model, layer_index, *filter_index, use_cuda=True)
        model = model.to(device)
            
        print("iter{}. {:.2f}% Filters remaining".format(it, 100*float(total_num_filters(model)) / number_of_filters))
        
        accuracy, loss, loss_cls, loss_offset = test(model, args.test_path)
        print('iter{}. without retrain :: accuracy: {:.4f} loss: {:.4f} cls loss: {:.4f} offset loss: {:.4f}'.format(it, accuracy, loss, loss_cls, loss_offset))
            
        print("iter{}. Fine tuning to recover from prunning iteration.. ".format(it))
        torch.cuda.empty_cache()
        train(model, path=args.train_path, epoch = 6)
            
        since = time.time()
        accuracy, loss, loss_cls, loss_offset = test(model, args.test_path)
        print('iter{}. after retrain :: accuracy: {:.4f} loss: {:.4f} cls loss: {:.4f} offset loss: {:.4f}'.format(it, accuracy, loss, loss_cls, loss_offset))
        print("iter{}. test time cost is {:.2f} s".format(it, time.time()-since))
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'pnet_weights_pruned_{}'.format(it)))
        torch.save(model, os.path.join(save_dir, 'pnet_prunned_{}'.format(it)))
        
    print("Finished prunning")