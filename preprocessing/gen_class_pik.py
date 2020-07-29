#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:24:48 2020
@author: sipo
INESCTEC
"""

import os
import sys
import pickle
import random
import numpy as np


data_path = ''   #path with .pkl files per IHC slide
classes = 2      #number of classes to combine (according IHC status, 2 classes, or IHC score, 4 classes)

save_path = os.path.join(data_path, str(classes) + '_classes')
if not os.path.exists(save_path):
    os.mkdir(os.path.join(data_path, str(classes) + '_classes'))

data_files = np.array([f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.pkl')])

if classes == 2:
    data_labels = np.array([f.split('_')[-1] for f in data_files])
else:
    data_labels = np.array([f.split('_')[-2] for f in data_files])

sum_tiles = []
aux_tiles = []

print('>>> READING TILES PICKLES <<<')
for ii in range(classes):
    class_files = data_files[np.where(data_labels == str(ii))]
    aux = 0
    
    aux2 = []
    for jj in range(len(class_files)):
        pik = open(os.path.join(data_path, data_files[jj] + '.pkl'), 'rb')
        tiles, _ , _, _ = pickle.load(pik)
        pik.close()
        aux += len(tiles)
        aux2.append(len(tiles))
        sys.stdout.write('\r' + 'class {}: ....pickle {}/{}'.format(ii, jj+1, len(class_files)))
        sys.stdout.flush()
    
    print('\n....total tiles: ', aux, '\n')
    
    sum_tiles.append(aux)
    aux_tiles.append(aux2)
    
to_save = min(sum_tiles)

print('>>> REARRANGING PICKLES INTO CLASSES <<<')
for ii in range(classes):
    class_files = data_files[np.where(data_labels == str(ii))]
    aux = 0

    patches = []
    for jj in range(len(class_files)): 
        pik = open(os.path.join(data_path, data_files[jj] + '.pkl'), 'rb')
        tiles, _ , _, _ = pickle.load(pik)
        pik.close()
        
        for l in range(len(tiles)):
            patches.append(tiles[l])
      
        sys.stdout.write('\r' + 'class {}: ....pickle {}/{}'.format(ii, jj+1, len(class_files)))
        sys.stdout.flush()
        
    aux = random.sample(range(len(patches)), to_save)
    patches = np.array(patches)[aux]

    print('\nsaving', len(patches), 'patches of class', ii, 'into .pkl file\n')
    f = open(os.path.join(save_path, 'class_' + str(ii)), 'wb') 
    pickle.dump([patches, len(patches), ii], f)
    f.close()
