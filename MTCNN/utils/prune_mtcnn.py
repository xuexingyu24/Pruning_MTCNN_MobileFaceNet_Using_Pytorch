#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:42:25 2019

@author: xingyu
"""

import torch
import numpy as np

def replace_layers(model, i, indexes, layers):

    """
    replace conv layers of model.feature

    :param model:
    :param i: index of model.feature
    :param indexes: array of indexes of layers to be replaced
    :param layers: array of new layers to replace
    :return: model with replaced layers
    """
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_Conv2d(conv, filter_index, Next=False, use_cuda = True):

    """
    :param conv: conv layer to be pruned
    :param filter_index: filter index to be pruned
    :param Next: False: the conv to be pruned by reconstructing the out_channel,
                 True: represent the next conv to be pruned by reconstructing the input_channel
    :param use_cuda:
    :return:
    """

    if Next:
        new_conv = \
            torch.nn.Conv2d(in_channels=conv.in_channels - len(filter_index), \
                            out_channels=conv.out_channels, \
                            kernel_size=conv.kernel_size, \
                            stride=conv.stride,
                            padding=conv.padding,
                            dilation=conv.dilation,
                            groups=conv.groups,
                            bias=(conv.bias is not None))

        old_weights = conv.weight.data.cpu().numpy() # i.e. (512, 512, 3, 3)
        new_weights = np.delete(old_weights, filter_index, axis=1)  # i.e. (512, 511, 3, 3)
        new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_conv.weight.data = new_conv.weight.data.cuda()

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

        bias_numpy = conv.bias.data.cpu().numpy()  # i.e.  (512,)
        bias = np.delete(bias_numpy, filter_index)  # i.e. (511,)
        new_conv.bias.data = torch.from_numpy(bias)
        if use_cuda:
            new_conv.bias.data = new_conv.bias.data.cuda()

    return new_conv

def prune_PReLu(prelu, filter_index, use_cuda=True):
    # prune PReLu
    new_prelu = torch.nn.PReLU(num_parameters=prelu.num_parameters - len(filter_index))
    old_weights = prelu.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filter_index)
    new_prelu.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_prelu.weight.data = new_prelu.weight.data.cuda()

    return new_prelu

def prune_linear(linear_layer, conv, filter_index, use_cuda=True):
    # prune fully connected layer which is the next to the to-be-pruned conv layer
    params_per_input_channel = linear_layer.in_features // conv.out_channels
    new_linear_layer = torch.nn.Linear(linear_layer.in_features - len(filter_index) * params_per_input_channel,
                                       linear_layer.out_features, bias=(linear_layer.bias is not None))

    old_weights = linear_layer.weight.data.cpu().numpy()  # i.e. (4096, 25088)  (out_feature x in_feature)

    delete_array = []
    for filter in filter_index:
        delete_array += [filter * params_per_input_channel + x for x in range(params_per_input_channel)]
        new_weights = np.delete(old_weights, delete_array, axis=1)  # i.e. (4096, 25039)

    new_linear_layer.weight.data = torch.from_numpy(new_weights)

    if linear_layer.bias is not None:
        new_linear_layer.bias.data = linear_layer.bias.data
    if use_cuda:
        new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

    return new_linear_layer


def prune_mtcnn(model, layer_index, *filter_index, use_cuda=False):
    
    _, conv = list(model.features._modules.items())[layer_index]
    _, prelu = list(model.features._modules.items())[layer_index+1]
    next_conv = None
    offset = 1

    if len(filter_index) >= conv.out_channels:
        raise BaseException("Cannot prune the whole conv layer")

    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1

    # The new conv layer constructed as follow:
    new_conv = prune_Conv2d(conv, filter_index, Next=False, use_cuda = use_cuda)
    
    # The new PReLU layer constructed as follow:    
    new_PReLU = prune_PReLu(prelu, filter_index, use_cuda = use_cuda)
        
    # next conv layer needs to be reconstructed
    if not next_conv is None:
        next_new_conv = prune_Conv2d(next_conv, filter_index, Next=True, use_cuda = use_cuda)

        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index+1, layer_index + offset], \
                             [new_conv, new_PReLU, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features  # reset
        del conv # reset

        model.features = features
    
    else:
        
        linear_layer = None
        offset = 1
        while layer_index + offset < len(model.features._modules.items()):
            res = list(model.features._modules.items())[layer_index + offset]
            if isinstance(res[1], torch.nn.Linear):
                layer_name, linear_layer = res
                break
            offset = offset + 1
        
        if not linear_layer is None:
            new_linear_layer = prune_linear(linear_layer, conv, filter_index, use_cuda = use_cuda)

            features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+1, layer_index + offset], \
                                 [new_conv, new_PReLU, new_linear_layer]) for i, _ in enumerate(model.features)))
            
            del model.features  # reset
            del conv # reset

            model.features = features
        
        else:
            model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+1], \
                                 [new_conv, new_PReLU]) for i, _ in enumerate(model.features)))
        
        if hasattr(model, 'conv4_1') and hasattr(model, 'conv4_2'):
        
            conv4_1, conv4_2 = model.conv4_1, model.conv4_2         
            # new conv4_1 and conv4_2 layer need to be reconstructed
            new_conv4_1 = prune_Conv2d(conv4_1, filter_index, Next=True, use_cuda = use_cuda)
            new_conv4_2 = prune_Conv2d(conv4_2, filter_index, Next=True, use_cuda = use_cuda)
            
            del model.conv4_1
            del model.conv4_2
            
            model.conv4_1 = new_conv4_1
            model.conv4_2 = new_conv4_2
    
    return model

if __name__ == '__main__':
    
    import sys
    sys.path.append("../Base_Model")
    
    from MTCNN_nets import PNet, RNet, ONet

    model = ONet()
    model.train()
    
    layer_index = 9
    filter_index = (2,4)
    
    model = prune_mtcnn(model, layer_index, *filter_index, use_cuda=False)

    print(model)
        

    
    

    
    

        
        
        
        