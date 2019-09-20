#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:46:37 2019

Class FilterPrunner performs structured pruning on filters based on the first order Taylor expansion of the network cost function from Nvidia
"Pruning Convolutional Neural Networks for Resource Efficient Inference" - arXiv:1611.06440

Args:
        model: the DNN model which should be composed with model.features and model.classifier.

@author: xingyu
"""
import torch
import numpy as np 
from operator import itemgetter
from heapq import nsmallest

class FilterPrunner:
    def __init__(self, model, use_cuda = False):
        self.model = model
        self.reset()
        self.use_cuda = use_cuda

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer # the ith conv2d layer
                activation_index += 1
                
        if hasattr(self.model, 'conv4_1') and hasattr(self.model, 'conv4_2'):
            a = self.model.conv4_1(x)
            b = self.model.conv4_2(x)
            c = None 
            
        if hasattr(self.model, 'conv5_1') and hasattr(self.model, 'conv5_2'):
            a = self.model.conv5_1(x)
            b = self.model.conv5_2(x)
            c = None 
        
        if hasattr(self.model, 'conv6_1') and hasattr(self.model, 'conv6_2') and hasattr(self.model, 'conv6_3'):
            a = self.model.conv6_1(x)
            b = self.model.conv6_2(x)
            c = self.model.conv6_3(x)

        return c, b, a

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        taylor = activation * grad

        # Get the average value for every filter,
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
                
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
    
    
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])

        return filters_to_prune_per_layer