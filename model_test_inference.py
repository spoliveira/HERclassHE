#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:24:05 2020
@author: sipo, jtrp & tfg
INESCTEC
"""

import os
import datetime
import torch
from torch import nn
from model_utils import get_flat_dim, ConvNet, test_model, inference

# --- Data directories ---
data_path = ''         #path with test/inference .pkl files (per slide) with HE tiles


# --- Hyperparameters ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'    #gpu to use
BATCH_TILE = 300                                             #number of tiles to select per slide (default: 300)
AGGREGATION = 'mean'                                         #aggregation method ('mlp', 'mean' or 'median')
cnn_file = ''                                                #file with model weights ****** WARNING: the aggregation method should be the same of the model ******
mode = 'test'                                                #'test' (if ground-truth labels available) or 'inference '(without ground-truth labels)


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
if os.path.exists(cnn_file):
    weights = torch.load(cnn_file, map_location=DEVICE)
    model.load_state_dict(weights['state_dict'], strict=False)
    model.to(DEVICE)

else:
    print('WARNING: model file does not exists!')

if mode == 'inference':
    predictions = inference(DEVICE, model, AGGREGATION, data_path, BATCH_TILE=BATCH_TILE)

elif mode == 'test':
    criterion = nn.CrossEntropyLoss()
    ACC, F1, precision, recall = test_model(DEVICE, model, AGGREGATION, criterion, data_path, BATCH_TILE=BATCH_TILE)
