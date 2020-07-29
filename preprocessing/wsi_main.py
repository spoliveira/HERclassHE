#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:09:16 2019
@author: sipo
INESCTEC
"""

import os
import time
import numpy as np
from datetime import timedelta
from wsi_utils import *


# --- Data directories ---
dir_slides = ''         #path with all the slides
dir_patches = ''        #path where .pkl files will be stored (one file per slide containing [patches, coord, label(IHC score), bin_label(IHC status)])
dir_thumbnails = ''     #path where thumbnails will be stored (img/otsu masks)
dir_masks = ''          #annotation masks location (if there are annotation mask files)
labels_file = ''        #.csv file with labels (structure: slide_name, IHC score (4 classes), IHC status (2 classes))


# --- Hyperparameters ---
slide_type = '.svs'     #slide file extension (.svs, .tif, .tiff, .ndpi, ...)
modality = 'HE'         #slide modality ('HE' or 'HER2')
patch_size = 256        #tile size to extract from slide with original dimensions (default: 256)
down_factor = 32        #slide reduction factor for thumbnail, otsu & tile processing (default: 32)
overlap = 0             #tiles overlap (default: 0)
sort_threshold = 0      #tile acceptance threshold (default: 0; only tiles 100% inside mask are accepted)

model_mode = 'train'    #'train':known labels; 'test':unknown labels
save_db = True          #save/not save tiles in .pkl file per slide (default: True)
save_patches = False    #save/not save tiles in .png (default: Fasle)
save_thumb = True       #save/not save slide and otsu thumbnails (default: True)

wsi_img = WSI(model_mode, slide_type, dir_slides, labels_file, dir_patches, dir_thumbnails, dir_masks)

start_time = time.time()
info = wsi_img.gen_db(modality, patch_size, down_factor, overlap, sort_threshold, SAVE_DB=save_db, SAVE_PATCHES=save_patches, SAVE_THUMB=save_thumb)
np.savetxt(os.path.join(dir_patches,'info_tiles_' + modality + '_' + model_mode + '.csv'), info, delimiter=",", fmt='%s')
end_time = time.time()

print('\nTotal time:',timedelta(seconds=int(round(end_time - start_time))))
