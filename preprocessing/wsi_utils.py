#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:57:23 2019
@author: sipo
INESCTEC
"""

import os
import sys
import time
from datetime import timedelta

import pickle
import numpy as np
from numpy import genfromtxt

from openslide import *
from openslide.deepzoom import DeepZoomGenerator

from matplotlib import pyplot as plt

import cv2
from PIL import Image, ImageOps
from skimage import filters

class WSI(object):
    
    def __init__(self, model_mode, slide_type, files_dir, labels_file, db_dir, thumb_dir, dir_masks):
        """
        - model_mode     train/ test (known labels/ unknown labels)
        - slides_type    slides file extension (.svs, .tif, .tiff, .ndpi, ...)
        - slides_dir     path with all the slides
        - labels_file    .csv file with labels
        - db_dir         path where .pickle files will be stored (one file per slide containing [patches, coord, label(IHC score), bin_label(IHC status)])
        - thumb_dir      path where thumbnails will be stored (img/otsu masks)
        - dir_masks      path with annotation masks (if there are annotation mask files)
        """
    
        self.model_mode = model_mode
        self.slide_type = slide_type
        self.files_dir = files_dir
        self.files = np.array([f for f in os.listdir(files_dir) if f.endswith(slide_type)])
        self.num_files = len(self.files)
        
        self.labels_info = genfromtxt(labels_file, delimiter=',', dtype='str') 
                
        self.db_loc = db_dir
        
        self.thumb_dir = thumb_dir
        self.img_dir = os.path.join(thumb_dir,'img')
        self.otsu_dir = os.path.join(thumb_dir,'otsu')
        
        self.mask_dir = dir_masks
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        elif not os.path.exists(self.otsu_dir):
            os.mkdir(self.otsu_dir)

        print("===============================================================================")
        print("Images directory: ", self.files_dir)
        print("Patches directory: ", self.db_loc)
        print(slide_type, "files found: ", self.num_files)
        print("===============================================================================")
           
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
    def gen_db(self, modality, patch_size, down_factor, overlap, threshold, SAVE_DB=False, SAVE_PATCHES=False, SAVE_THUMB=False):
                        
        init = 0
        to_remove = []
        info = []
        
        for ii in range(self.num_files):

            start_time = time.time()
            RE_DO = True
        
            tile_size = patch_size - 2*overlap            
            file_name = self.files[ii]   
            file_name = file_name.split('.')[0]                
            
            print()
            print(ii+1,'/',self.num_files)
            print(file_name)
            
            label, bin_label = self.check_svs_label(file_name)

            if self.model_mode == 'train' and (label == -1 or bin_label == -1):
                print('WARNING: svs file without label --> not used')
                continue

            print('>>>>>> Opening slide')
            slide = open_slide(os.path.join(self.files_dir, file_name + self.slide_type))
            file_dims = slide.level_dimensions[0]
            print('file dimensions ', file_dims)
            
            if SAVE_THUMB:
                print('>>>>>> Generating thumbnails')
                
                if self.slide_type == 'tif':
                    img = slide.read_region((0,0), 0, file_dims)
                    red_width = int(file_dims[0]/down_factor)
                    red_length = int(file_dims[1]/down_factor)
                    thumbnail = np.array(img.resize((red_width, red_length), Image.ANTIALIAS))

                else:
                    thumbnail = reduce_im(slide, file_dims, down_factor)
                
                print('>>>>>> Applying Otsu threshold')
                thumb_otsu = apply_otsu(thumbnail, modality, label)
                
                if not os.path.exists(os.path.join(self.otsu_dir,'otsu_over_thumb')):
                    os.mkdir(os.path.join(self.otsu_dir,'otsu_over_thumb'))

                _, contours, _ = cv2.findContours(np.uint8(thumb_otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                otsu_over = cv2.drawContours(np.asarray(thumbnail), contours, -1, (0, 0, 255), 3) 

                print('>>>>>> Saving thumbnails')
                save_file(thumbnail, file_name, self.img_dir)
                save_file(thumb_otsu, file_name, self.otsu_dir)
                save_file(otsu_over, file_name, os.path.join(self.otsu_dir,'otsu_over_thumb'))
                

            if not SAVE_THUMB:
                if not os.path.exists(os.path.join(self.otsu_dir, file_name +'.png')):
                    print('WARNING: no otsu mask file')
                    continue
                else:
                    print('>>>>>> Reading Otsu mask')    
                    thumb_otsu = Image.open(os.path.join(self.otsu_dir, file_name +'.png'), mode='r')
                
            
            if not os.path.exists(os.path.join(self.mask_dir, file_name +'.png')) and self.model_mode == 'train':
                print('WARNING: no annotation file')

            else:
                if self.model_mode == 'train':
                    print('>>>>>> Reading annotation mask & combining with otsu mask')
                    thumb_mask = Image.open(os.path.join(self.mask_dir, file_name +'.png'), mode='r')

                    a = thumb_mask.size
                    b = thumb_otsu.size
                    
                    if  a != b:
                        padding = (0, 0, b[0]-a[0], b[1]-a[1])
                        thumb_mask= ImageOps.expand(thumb_mask, padding, fill=25)

                    thumb_combined = cv2.bitwise_and(np.uint8(thumb_otsu), np.uint8(thumb_mask))
                    thumb_combined_ = Image.fromarray(thumb_combined*255)
                    
                    if SAVE_THUMB:
                        if not os.path.exists(os.path.join(self.otsu_dir,'combined_masks')) & os.path.exists(os.path.join(self.otsu_dir,'combined_over_thumb')):
                            os.mkdir(os.path.join(self.otsu_dir,'combined_masks'))
                            os.mkdir(os.path.join(self.otsu_dir,'combined_over_thumb'))
                        
                        save_file(thumb_combined_, file_name, os.path.join(self.otsu_dir,'combined_masks'))

                        _, cnts, _ = cv2.findContours(thumb_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                        msk_over = cv2.drawContours(np.asarray(thumbnail), cnts, -1, (0, 0, 0), 3) 
                        save_file(msk_over, file_name, os.path.join(self.otsu_dir,'combined_over_thumb'))
            
            
            while RE_DO:
                print('>>>>>> Generating patches')
                tiles_svs = DeepZoomGenerator(slide, tile_size, overlap)  
                
                if os.path.exists(os.path.join(self.mask_dir, file_name +'.png')) and self.model_mode == 'train':
                    tiles_msk = DeepZoomGenerator(ImageSlide(thumb_combined_), np.around(tile_size/down_factor), np.around(overlap/down_factor))  
                
                else:
                    tiles_msk = DeepZoomGenerator(ImageSlide(thumb_otsu), np.around(tile_size/down_factor), np.around(overlap/down_factor))   
                                      
                print('>>>>>> Sorting patches')
                patches, loc, c = self.sort_patches(file_name, label, bin_label, tiles_svs, tiles_msk, patch_size, SAVE_DB, SAVE_PATCHES)

                if modality == 'HER2':
                    RE_DO = False
                    overlap = 0
                    
                elif c >= 4500:
                    RE_DO = False
                    overlap = 0
                
                elif c > 3375 and c < 4500:
                    overlap = int(0.3 * patch_size/2)
                    tile_size = patch_size - 2 * overlap
                
                else:
                    overlap = int((1 - np.around(c/4500, decimals=1)) * patch_size/2)
                    tile_size = patch_size - 2 * overlap
                
            final = init + c - 1
            
            if SAVE_DB:
                print('>>>>>> Saving patches into pickle file')
                if self.model_mode == 'test':
                    f = open(os.path.join(self.db_loc, file_name + '.pkl'), 'wb') 
                else:   
                    f = open(os.path.join(self.db_loc, file_name + '_' + str(label) + '_' + str(bin_label) + '.pkl'), 'wb') 

                pickle.dump([patches, loc, label, bin_label], f)
                f.close()
                                
            
            info.append(np.array([file_name, c, init, final, label, bin_label]))
            init = final + 1
            end_time = time.time()
            print('elapsed time:',timedelta(seconds=int(round(end_time - start_time))),'\n')
         
        print(to_remove)
        return info
        
        
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def sort_patches(self, file_name, label, bin_label, tiles_svs, tiles_combined, patch_size, SAVE_DB=False, SAVE_PATCHES=False):
                
        max_level_svs = tiles_svs.level_count - 1        
        max_level_otsu = tiles_combined.level_count - 1 
        
        x_tiles, y_tiles = tiles_combined.level_tiles[max_level_otsu]
        x, y = 0, 0
        count = 0
        patches, loc = [], []
        
        while y < y_tiles:
            while x < x_tiles:                                           
                new_tile_combined = np.array(tiles_combined.get_tile(max_level_otsu, (x, y)))

                if (np.sum(new_tile_combined, axis=2) == 0).sum() == 0:
                    try:
                        new_tile_svs = np.array(tiles_svs.get_tile(max_level_svs, (x, y)), dtype=np.uint8)

                        if new_tile_svs.shape == (patch_size, patch_size, 3):
                            count += 1
                        
                            if SAVE_DB:
                                hls = cv2.cvtColor(new_tile_svs, cv2.COLOR_RGB2HLS)
                                _, l, _ = cv2.split(hls)                        
                                patches.append(l)
                                loc.append(np.array([x, y]))
                            
                            if SAVE_PATCHES:
                                self.save_patches(file_name, new_tile_svs, np.array([x, y]), label, bin_label)
                    except:
                        print('Warning')
                                                       
                x += 1              
            y += 1
            x = 0
            sys.stdout.write("\r" + '.......... {}/{} patches lines'.format(y, y_tiles))
        
        print('\npatches to save:', count)
        return patches, loc, count   

    
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def save_patches(self, file_name, patch, loc, label, bin_label):
                
        save_dir = self.db_loc
    
        if self.model_mode == 'test':
            png_name = file_name + '_' + str(loc[0]) + '_' + str(loc[1]) + '.png'

        else:
            png_name = file_name + '_L' + str(label) + '_L' + str(bin_label) + '_' + str(loc[0]) + '_' + str(loc[1]) + '.png'

        if not os.path.exists(os.path.join(save_dir, file_name)):
            os.mkdir(os.path.join(save_dir, file_name))
        
        Image.fromarray(np.uint8(patch)).save(os.path.join(save_dir, file_name, png_name))

                
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
    def check_svs_label(self, file_name):
        
        loc = np.where(self.labels_info[:,0] == file_name)[0]
        
        if len(loc) == 1:
            if not self.labels_info[loc, 1]:
                label = -1
            else:
                label = self.labels_info[loc, 1][0].astype('int')
            
            if not self.labels_info[loc, 2]:
                bin_label = -1
            else:
                bin_label = self.labels_info[loc, 2][0].astype('int')
               
        else:
            label = -1
            bin_label = -1
            
        return label, bin_label
        
        
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def reduce_im(slide, file_dims, down_factor=32):
    
    thumbnail = slide.get_thumbnail((int(np.floor(file_dims[0]/down_factor)),int(np.floor(file_dims[1]/down_factor))))
    
    return thumbnail


#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def apply_otsu(thumbnail, modality, label=''):
    
    rgb = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGBA2RGB)
    hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_BGR2HSV)

    if modality == 'HE' or (modality == 'HER2' and (label == 2 or label == 3)):
        _, s, _ = cv2.split(hsv)
        s = cv2.GaussianBlur(s ,(15,15),0)
        val = filters.threshold_otsu(s)
        _, thumb_otsu = cv2.threshold(s , val, 255, cv2.THRESH_BINARY)   

    else:
        _, _, v = cv2.split(hsv)
        v = cv2.GaussianBlur(v ,(15,15),0)
        thumb_otsu = (v < np.max(v) - 5) * 255  

    thumbnail_otsu = Image.fromarray(np.uint8(thumb_otsu))
    
    return thumbnail_otsu   


#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def save_file(image, svs_name, dir_save):
    
    img_name = svs_name + '.png'
    out = Image.fromarray(np.uint8(image))
    out.save(os.path.join(dir_save,img_name))
