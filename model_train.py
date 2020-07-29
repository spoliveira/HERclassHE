#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:56:22 2020
@author: sipo, jtrp & tfg
INESCTEC
"""

import os
import datetime
import torch
from torch import nn
from model_utils import get_flat_dim, ConvNet, train_model

# --- Data directories ---
train_path = ''        #path with train .pkl files (per slide) with HE tiles
val_path = ''          #path with validation .pkl files (per slide) with HE tiles


# --- Hyperparameters ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #gpu to use
N_EPOCHS = 150                                              #number of training epochs  (default: 150)
BATCH_TILE = 300                                            #number of tiles to select per slide (default: 300)
IMGS_STEP = 2                                               #number of slides to accumulate after backprop (default: 2)
AGGREGATION = 'mlp'                                         #aggregation method ('mlp', 'mean' or 'median')
PRETRAIN = True                                             #use/not use IHC-HER2 pretrained CNN weights
cnn_file = './authors_models/IHC-HER2_model.tar'            #file with IHC-HER2 pretrained CNN weights


# --- CNN architecture ---
INPUT_DIM = [1, 256, 256]   
N_CONV = 4
N_POOL = 4
CONV_FILTERS = [16, 32, 64, 128]
K_SIZES = [5,3,3,3]
P_KERNELS = [2,2,2,2]
STRIDES = [1,1,1,1]
P_STRIDES = [2,2,2,2]
PADDINGS = [0,0,0,0]
N_FC = 3
FC_DIMS = [1024, 256, 4]
N_MLP = 4
MLP_DIMS = [256, 128, 64, 2]

flat_dim = get_flat_dim(INPUT_DIM, N_CONV, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS)
model = ConvNet(N_CONV, N_POOL, N_FC, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS, FC_DIMS, 
                N_MLP, MLP_DIMS, BATCH_TILE, INPUT_DIM[0], flat_dim, DEVICE).to(DEVICE)


# --- IHC pretrained weights ---
time = datetime.datetime.now().replace(second=0, microsecond=0)
if PRETRAIN == True:

    if os.path.exists(cnn_file):
        weights = torch.load(cnn_file, map_location=DEVICE)
        model.load_state_dict(weights['state_dict'], strict=False)
        model.to(DEVICE)
        filename = 'model_{}_IHCweights'.format(time)

        print('IHC pretrained weights loaded') 
    
    else:
        print('No model file! IHC pretrained weights not loaded --> Random initialization')
        filename = 'model_{}_randomweights'.format(time)

else:
    filename = 'model_{}_randomweights'.format(time)


# --- Training/validation loop ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_hist, val_hist, val_acc, val_F1 = train_model(DEVICE, filename, model, AGGREGATION, criterion, optimizer,
                                                    N_EPOCHS, IMGS_STEP, train_path, val_path, BATCH_TILE=BATCH_TILE)
