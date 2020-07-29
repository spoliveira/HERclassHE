#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:36:31 2019
@author: sipo
INESCTEC
"""

import os
import sys
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


def get_flat_dim(input_dim, n_conv, conv_filters, kernel_sizes, p_kernel, strides, p_strides, paddings):
    _, H, W = input_dim
    for i in range(n_conv):
        H = int(math.floor((H + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        H = int(math.floor((H - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
        W = int(math.floor((W + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        W = int(math.floor((W - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
    
    flat_dim = H * W * conv_filters[-1]
    
    return flat_dim
    
    
class ConvNet(nn.Module):
    def __init__(self, n_conv, n_pool, n_fc, conv_filters, kernel_sizes, p_kernels, strides, p_strides,
                 paddings, fc_dims, in_channels, flat_dim, dropout=.0, batch_norm=False):

        super(ConvNet, self).__init__()
        
        self.n_conv = n_conv              # integer
        self.n_pool = n_pool              # integer
        self.n_fc = n_fc                  # integer
        self.conv_filters = conv_filters  # list with length n_conv
        self.kernel_sizes = kernel_sizes  # list with length n_conv (square filters)
        self.p_kernels = p_kernels        # list with length n_pool (square filters)
        self.strides = strides            # list with length n_conv
        self.p_strides = p_strides        # list with length n_pool
        self.paddings = paddings          # list with length n_conv
        self.fc_dims = fc_dims            # list with length n_fc
        self.in_channels = in_channels    # integer
        self.flat_dim = flat_dim          # integer
        
        # convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(self.in_channels, 
                                                    self.conv_filters[0], 
                                                    self.kernel_sizes[0], 
                                                    stride=self.strides[0], 
                                                    padding=self.paddings[0])])
        
        self.conv_layers.extend([nn.Conv2d(self.conv_filters[i-1],
                                           self.conv_filters[i],
                                           self.kernel_sizes[i],
                                           stride=self.strides[i], 
                                           padding=self.paddings[i])
                                 for i in range(1, self.n_conv)])

        # pooling layers
        self.pool_layers = nn.ModuleList([nn.LPPool2d(2,
                                                      self.p_kernels[0],
                                                      stride=self.p_strides[0])])
        
        self.pool_layers.extend([nn.LPPool2d(2,
                                             self.p_kernels[i],
                                             stride=self.p_strides[i])
                                 for i in range(1, self.n_pool)])
    
        # fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.flat_dim, self.fc_dims[0])])
        self.fc_layers.extend([nn.Linear(self.fc_dims[i-1], self.fc_dims[i]) for i in range(1, self.n_fc)])                                            
                                                     
                                                     
    def forward(self, X, get_activations=False):
        
        activations = []
        N = X.shape[0]
        
        # forward pass through the conv. layers
        h = X
        for i in range(self.n_conv):
            h = self.conv_layers[i](h)
            h = F.relu(h)                                                                            
            activations.append(h)
            h = self.pool_layers[i](h)
            
        # flatten the activation before applying the fc layers
        h = h.reshape(N, -1)
        
        # forward pass through the fc layers
        for i in range(self.n_fc-1):
            h = self.fc_layers[i](h)
            h = F.relu(h)                                         
            activations.append(h)                                         
            
        # the output layer does not have ReLU, batch norm. and dropout, so we leave it outside the loop
        y = self.fc_layers[self.n_fc-1](h)
        
        if get_activations:
            return y, activations
        else:
            return y
    
    def predict(self, X): #Computes the probabilities of each class for each example in X. 
        
        logits = self.forward(X)
        probs = F.softmax(logits, dim=1)
        
        return probs
