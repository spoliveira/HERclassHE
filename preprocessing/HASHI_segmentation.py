#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:40:05 2019
@author: sipo
INESCTEC
"""

import os
import sys
import cv2
import time
from datetime import timedelta

import ghalton
import numpy as np

from PIL import Image
from scipy import interpolate

from openslide import *
from openslide.deepzoom import DeepZoomGenerator

import torch
import torchvision.transforms.functional as TF

from HASHI_cnn_utils import ConvNet, get_flat_dim


# --- Data directories ---
rootp = os.path.dirname(os.path.abspath(__file__))
dir_slides = ''                 #path with all the slides
dir_thumbnails = ''             #path where segmentation masks will be stored (reduced size)


# --- Hyperparameters ---
slide_type = '.svs'   #slide file extension (.svs, .tif, .tiff, .ndpi, ...)
down_factor = 32      #slide reduction factor for otsu & tile processing
tile_size = 512       #tile size to extract from slide with original dimensions

T = 20                #number of iterations (HASHI method)
N = 100               #samples per iteration (HASHI method)
threshold = 0.24      #probability threshold to convert probability map into segmentation mask


# --- CNN architecture ---
N_CLASSES = 2
INPUT_DIM = [3, 101, 101] 
N_CONV = 1
N_POOL = 1
CONV_FILTERS = [256]
K_SIZES = [8]
P_KERNELS = [2]
STRIDES = [1]
P_STRIDES = [2]
PADDINGS = [0]
N_FC = 2
FC_DIMS = [256, N_CLASSES]

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
flat_dim = get_flat_dim(INPUT_DIM, N_CONV, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS)
model = ConvNet(N_CONV, N_POOL, N_FC, CONV_FILTERS, K_SIZES, P_KERNELS, STRIDES, P_STRIDES, PADDINGS, FC_DIMS, INPUT_DIM[0], flat_dim).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(rootp, 'HASHI_trained_cnn_model.pth'), map_location=DEVICE))
model.eval()


# --- HASHI method ---
wsi_list = np.array([f for f in os.listdir(dir_slides) if f.endswith(slide_type)])

rgb2yuv = (0.299, 0.587, 0.114, 0, -0.147, -0.289, 0.436, 0, 0.615, -0.515, -0.100, 0)

for idx in range(len(wsi_list)):

    start = time.time()
    filename = wsi_list[idx].split('.')[0]
    print(filename)
    
    wsi_file = os.path.join(dir_slides, wsi_list[idx])
    try:
        thumb_file = os.path.join(dir_thumbnails + '/img', filename + '.png')

        wsi = open_slide(wsi_file)
        wsi_dims = wsi.level_dimensions[0]
        
        thumbnail = Image.open(thumb_file).convert('RGB')
        thumbw, thumbh = thumbnail.size

    except:
        print('.... NO FILE ....\n')
        continue
    
    # --- Map initialization ---#
    grid = np.zeros([thumbw, thumbh])
    x = np.arange(0, grid.shape[0])
    y = np.arange(0, grid.shape[1])
    xx, yy = np.meshgrid(x, y)
    
    values = np.array([])
    total_points = np.empty((0,2))
    total_points_svs = np.empty((0,2))
    sequencer = ghalton.Halton(2)
    
    for ii in range(T):    
        # --- Pseudo-random tile generator ---#    
        if ii==0:
            points = sequencer.get(N)
            points_thumb = np.floor(points * np.array([thumbw, thumbh])).astype('int')
            points_wsi = points_thumb * 32 - int(np.round(tile_size/2))
            
        else:
            points = sequencer.get(3*N)
            points_thumb = np.floor(points * np.array([thumbw, thumbh])).astype('int')
            
            points_dg = dg[points_thumb[:,1], points_thumb[:,0]]
            aux = np.flip(np.argsort(points_dg), 0)
            points_thumb = points_thumb[aux[:N]]
            points_wsi = points_thumb * 32 - int(np.round(tile_size/2))
    
        it_values = np.array([])
        for j in range(N):
            total_points = np.append(total_points, points_thumb[j].reshape([-1,2]), axis=0)
            total_points_svs = np.append(total_points_svs, points_wsi[j].reshape([-1,2]), axis=0)
            tile = wsi.read_region((points_wsi[j][0], points_wsi[j][1]), 0, (tile_size, tile_size))
            
            tile_RGB = tile.convert('RGB')
            tile_yuv = tile_RGB.convert('RGB',rgb2yuv)
            
            tile_ = np.array(tile_yuv.resize((101,101), Image.ANTIALIAS))
            tile_ = np.reshape(tile_, (101,101,3))
           
            tileT = TF.to_tensor(tile_)
        
            with torch.no_grad():
                pred =  model.predict(tileT.unsqueeze(0).to(DEVICE)).cpu().detach().numpy()
                it_values = np.append(it_values, np.array(pred[0,1]))
               
        values = np.append(values, it_values)
        f = interpolate.griddata(total_points, values, (xx,yy), method='cubic')
        f[np.isnan(f)]=0
        f[f<0]=0
        f[f>1]=1
        f1 = np.copy(f)
        f1[f1<threshold] = 0
        f1[f1>=threshold] = 1
        
        g = np.gradient(f)
        dg = np.sqrt(g[0]**2 + g[1]**2)
        
        sys.stdout.write("\r" + '........................{}/{} steps'.format(ii, T))

    print('\nelapsed time:',timedelta(seconds=int(round(time.time() - start))))
    
    print('>>> Saving results\n')
    if not os.path.exists(os.path.join(dir_thumbnails,'HASHI_msk')):
        os.mkdir(os.path.join(dir_thumbnails,'HASHI_msk'))

    if not os.path.exists(os.path.join(dir_thumbnails,'HASHI_msk','HASHI_over_thumb')):
        os.mkdir(os.path.join(dir_thumbnails,'HASHI_msk','HASHI_over_thumb'))

    msk_file = os.path.join(dir_thumbnails,'HASHI_msk', filename + '.png')
    img_file = os.path.join(dir_thumbnails,'HASHI_msk','HASHI_over_thumb', filename + '.png')

    out = Image.fromarray((f1*255).astype('uint8'))
    out.save(msk_file)

    _, contours, _ = cv2.findContours(f1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(np.asarray(thumbnail), contours, -1, (0, 0, 0), 4)
    img = Image.fromarray(img)
    img.save(img_file)
