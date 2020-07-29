#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:36:31 2019
@author: sipo
--- Special Topics ---
"""

import datetime
import torch
from torch import nn
from cnn_ihc_utils import get_flat_dim, CNN_IHC, train_model


# --- Data directories ---
train_path = ''       #path with train .pkl files (per class) with IHC tiles
val_path = ''         #path with validation .pkl files (per class) with IHC tiles
 

# --- Hyperparameters ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  #gpu to use
N_EPOCHS = 200                                             #number of training epochs (default: 200)
BATCH_TILE = 128                                           #number of tiles per batch (default: 128)


# --- CNN architecture ---
N_CLASSES = 4
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


# --- CNN ---
flat_dim = get_flat_dim(INPUT_DIM, N_CONV, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS)
model = CNN_IHC(N_CONV, N_POOL, N_FC, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS, FC_DIMS, INPUT_DIM[0], flat_dim).to(DEVICE)


# --- Train Model & Save ---
time = datetime.datetime.now().replace(second=0, microsecond=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_hist, valid_hist, valid_acc = train_model(DEVICE, str(time), model, criterion, optimizer, N_EPOCHS, train_path, val_path, BATCH_TILE=BATCH_TILE)
