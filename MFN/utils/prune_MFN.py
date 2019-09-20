#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:14:09 2019

The code is intended to prune the mobilefacenet model with given layer_index and filter_index 
step 1. convert the model structure to a well organized module dict
step 2. prune the all Conv_block but not conv_dw
step 3. prune whole Residual block based on its output -- layer: 16, 37, 46 

@author: xingyu
"""
import torch
import numpy as np

def prune_Conv2d(conv, filter_index, Next=False, use_cuda = True):
    
    if conv.groups == conv.in_channels:       
        new_conv = \
            torch.nn.Conv2d(in_channels=conv.in_channels - len(filter_index), \
                            out_channels=conv.out_channels - len(filter_index),
                            kernel_size=conv.kernel_size, \
                            stride=conv.stride,
                            padding=conv.padding,
                            dilation=conv.dilation,
                            groups=conv.groups - len(filter_index),
                            bias=(conv.bias is not None))
        
        old_weights = conv.weight.data.cpu().numpy()  # i.e. (64, 1, 3, 3)
        new_weights = np.delete(old_weights, filter_index, axis=0)  # i.e. (63, 1, 3, 3)
        new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_conv.weight.data = new_conv.weight.data.cuda()
        
        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy() 
            bias = np.delete(bias_numpy, filter_index) 
            new_conv.bias.data = torch.from_numpy(bias)
            if use_cuda:
                new_conv.bias.data = new_conv.bias.data.cuda()
        
    else: 
        if Next:
            new_conv = \
                torch.nn.Conv2d(in_channels=conv.in_channels - len(filter_index), \
                                out_channels=conv.out_channels,
                                kernel_size=conv.kernel_size, \
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                bias=(conv.bias is not None))
            
            old_weights = conv.weight.data.cpu().numpy()  # i.e. (512, 512, 3, 3)
            new_weights = np.delete(old_weights, filter_index, axis=1)  # i.e. (512, 511, 3, 3)
            new_conv.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                new_conv.weight.data = new_conv.weight.data.cuda()
                
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data  # bias is not changed

        else:
            new_conv = \
                torch.nn.Conv2d(in_channels=conv.in_channels, \
                                out_channels=conv.out_channels - len(filter_index),
                                kernel_size=conv.kernel_size, \
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                bias=(conv.bias is not None))
            
            old_weights = conv.weight.data.cpu().numpy()  # i.e. (512, 512, 3, 3)
            new_weights = np.delete(old_weights, filter_index, axis=0)  # i.e. (511, 512, 3, 3)
            new_conv.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                new_conv.weight.data = new_conv.weight.data.cuda()
                
            if conv.bias is not None:
                bias_numpy = conv.bias.data.cpu().numpy() 
                bias = np.delete(bias_numpy, filter_index) 
                new_conv.bias.data = torch.from_numpy(bias)
                if use_cuda:
                    new_conv.bias.data = new_conv.bias.data.cuda()
    
    return new_conv

def prune_BN(bn, filter_index, use_cuda = True):
    
    new_bn = torch.nn.BatchNorm2d(num_features = bn.num_features - len(filter_index))
    old_weights = bn.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filter_index)
    new_bn.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_bn.weight.data = new_bn.weight.data.cuda()
    
    return new_bn

def prune_PReLu(prelu, filter_index, use_cuda=True):

    new_prelu = torch.nn.PReLU(num_parameters=prelu.num_parameters-len(filter_index))
    old_weights = prelu.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filter_index)
    new_prelu.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_prelu.weight.data = new_prelu.weight.data.cuda()
        
    return new_prelu

def prune_linear(linear_layer, conv, filter_index, use_cuda=True):
    
    params_per_input_channel = linear_layer.in_features // conv.out_channels
    new_linear_layer = torch.nn.Linear(linear_layer.in_features - len(filter_index)*params_per_input_channel,linear_layer.out_features, bias=(linear_layer.bias is not None))

    old_weights = linear_layer.weight.data.cpu().numpy()  #i.e. (4096, 25088)  (out_feature x in_feature)

    delete_array = []
    for filter in filter_index:
        delete_array += [filter * params_per_input_channel + x for x in range(params_per_input_channel)]
        new_weights = np.delete(old_weights, delete_array, axis=1)  #i.e. (4096, 25039)

    new_linear_layer.weight.data = torch.from_numpy(new_weights)
    
    if linear_layer.bias is not None:
        new_linear_layer.bias.data = linear_layer.bias.data
    if use_cuda:
        new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()
        
    return new_linear_layer

def prune_MFN(model, layer_index, *filter_index, use_cuda=True):
    
    # regroup the model modules 
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

    if layer_index == None or filter_index == []:
        return model, modules
            
    if isinstance(modules[layer_index], Conv_block):
        if modules[layer_index].conv.groups != modules[layer_index].conv.in_channels:
        
            conv = modules[layer_index].conv
            bn = modules[layer_index].bn
            prelu = modules[layer_index].prelu
            
            modules[layer_index].conv = prune_Conv2d(conv, filter_index, Next=False, use_cuda = use_cuda)
            modules[layer_index].bn = prune_BN(bn, filter_index, use_cuda = use_cuda)
            modules[layer_index].prelu = prune_PReLu(prelu, filter_index, use_cuda = use_cuda)
            
            next_conv = modules[layer_index+1].conv
            modules[layer_index+1].conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)
            while modules[layer_index+1].conv.groups != 1:
                bn = modules[layer_index+1].bn
                modules[layer_index+1].bn = prune_BN(bn, filter_index, use_cuda = use_cuda)
                if isinstance(modules[layer_index+1], Conv_block):
                    prelu = modules[layer_index+1].prelu
                    modules[layer_index+1].prelu = prune_PReLu(prelu, filter_index, use_cuda = use_cuda)
                layer_index += 1
                if isinstance(modules[layer_index+2], Linear):
                    next_linear = modules[layer_index+2]
                    modules[layer_index+2] = prune_linear(next_linear, next_conv, filter_index, use_cuda = use_cuda)
                else:
                    next_conv = modules[layer_index+1].conv
                    modules[layer_index+1].conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)
                if isinstance(modules[layer_index+1], Flatten):
                    break
    
    if layer_index == 16:
        
        num_blocks = 4
        for i in range(num_blocks+1):
            conv = modules[layer_index - 3*i].conv
            bn = modules[layer_index - 3*i].bn
            
            modules[layer_index - 3*i].conv = prune_Conv2d(conv, filter_index, Next=False, use_cuda = use_cuda)
            modules[layer_index - 3*i].bn = prune_BN(bn, filter_index, use_cuda = use_cuda)
            
            next_conv = modules[layer_index+1-3*i].conv
            modules[layer_index+1-3*i].conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)
            
    if layer_index == 37:
        
        num_blocks = 6
        for i in range(num_blocks+1):
            conv = modules[layer_index - 3*i].conv
            bn = modules[layer_index - 3*i].bn
            
            modules[layer_index - 3*i].conv = prune_Conv2d(conv, filter_index, Next=False, use_cuda = use_cuda)
            modules[layer_index - 3*i].bn = prune_BN(bn, filter_index, use_cuda = use_cuda)
            
            next_conv = modules[layer_index+1-3*i].conv
            modules[layer_index+1-3*i].conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)
    
    if layer_index == 46:
        
        num_blocks = 2
        for i in range(num_blocks+1):
            conv = modules[layer_index - 3*i].conv
            bn = modules[layer_index - 3*i].bn
            
            modules[layer_index - 3*i].conv = prune_Conv2d(conv, filter_index, Next=False, use_cuda = use_cuda)
            modules[layer_index - 3*i].bn = prune_BN(bn, filter_index, use_cuda = use_cuda)
            
            next_conv = modules[layer_index+1-3*i].conv
            modules[layer_index+1-3*i].conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)
             
    index = 0
    for names, module in list(model._modules.items()):
        if isinstance(module, Depth_Wise):
            for _, module_sub in list(module._modules.items()):
                module_sub = modules[index]
                index += 1
                
        elif isinstance(module, Residual):
            for i in range(len(module.model)):
                for _, model_sub_sub in list(module.model[i]._modules.items()):
                    model_sub_sub = modules[index]
                    index += 1          
        else:
            model._modules[names] = modules[index]
            index += 1
    
    return model, modules

if __name__ == "__main__":

    import sys

    sys.path.append('..')
    from Base_Model.face_model import *
    
    model = MobileFaceNet(512)
    model.load_state_dict(torch.load('../Base_Model/MobileFace_Net', map_location=lambda storage, loc: storage))
           
    layer_index = 16
    filter_index = (2,4)
    
    model, module = prune_MFN(model, layer_index, *filter_index, use_cuda=False)
    print(module)
    print(model)
    
        
    
    
    
    
    
    



                
            
    

