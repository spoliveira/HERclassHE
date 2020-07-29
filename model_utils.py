#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:56:22 2020
@author: sipo, jtrp & tfg
INESCTEC
"""

import os
import sys
import math
import time
import pickle
import random
import numpy as np
from PIL import Image
from datetime import timedelta
import sklearn.metrics as metrics
from sklearn import preprocessing

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#--------------------------------------------------- Dataset -----------------------------------------------------
class dataset():
    def __init__(self, path_img, filename, lst=None, transform=None, inference=False):
        """
        Args:
            path_img (string):   path to .pkl files with tiles of each WSI image
            filename (string):   .pkl filename
            lst (list):          selected tiles indexes 
            transform (callable, optional): optional transform to be applied on a sample
            inference (boolean): model inference mode, no labels returned
        """    
    
        self.path_img = path_img
        
        with open(path_img, 'rb') as hf:
            patches, _, _, _ = pickle.load(hf)

        self.patches = patches

        self.inference = inference
        if self.inference == False:
            self.target = torch.tensor(int(filename.split('_')[-1]))
        else:
            self.target = ''

        if lst is not None:
            if torch.is_tensor(lst):
                self.patches = np.array(patches)[lst.cpu()]
            else:
                self.patches = np.array(patches)[lst]
        
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        a = Image.fromarray(self.patches[idx])
        tile = np.array(a.resize((256,256), Image.ANTIALIAS))
        tile = np.reshape(tile, (256,256,1))
        
        tile = self.transform(tile) 
        label = self.target

        if self.inference == False:
            return tile, label
        
        else:
            return tile

#-------------------------------------------------- CNN model ----------------------------------------------------
def get_flat_dim(input_dim, n_conv, conv_filters, kernel_sizes, p_kernel, strides, p_strides, paddings):

    _, H, W = input_dim
    for i in range(n_conv):
        H = int(math.floor((H + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        H = int(math.floor((H - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
        W = int(math.floor((W + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        W = int(math.floor((W - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
    
    flat_dim = H * W * conv_filters[-1]
    
    return flat_dim

    
def softargmax(y, device):

    beta = 1000
    softmax = F.softmax(y * beta, dim=1)
    pos = torch.arange(0, y.shape[1], 1).float()
    softargmax = torch.sum(softmax * pos.to(device), dim=1)
    return softargmax

        
class ConvNet(nn.Module):
    def __init__(self, n_conv, n_pool, n_fc, conv_filters, kernel_sizes, p_kernels, strides, p_strides, paddings,
                fc_dims, n_mlp, mlp_dims, select, in_channels, flat_dim, device):

        super(ConvNet, self).__init__()
        
        self.in_channels = in_channels    # integer
        self.n_conv = n_conv              # integer
        self.n_pool = n_pool              # integer
        self.conv_filters = conv_filters  # list with length n_conv
        self.kernel_sizes = kernel_sizes  # list with length n_conv (square filters)
        self.p_kernels = p_kernels        # list with length n_pool (square filters)
        self.strides = strides            # list with length n_conv
        self.p_strides = p_strides        # list with length n_pool
        self.paddings = paddings          # list with length n_conv
        self.n_fc = n_fc                  # integer
        self.fc_dims = fc_dims            # list with length n_fc
        self.n_mlp = n_mlp
        self.mlp_dims = mlp_dims
        self.select = select

        self.flat_dim = flat_dim          
        self.device = device
        
        # convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(self.in_channels, self.conv_filters[0], self.kernel_sizes[0], 
                                                    stride=self.strides[0], 
                                                    padding=self.paddings[0])])
        
        self.conv_layers.extend([nn.Conv2d(self.conv_filters[i-1], self.conv_filters[i], self.kernel_sizes[i],
                                           stride=self.strides[i], 
                                           padding=self.paddings[i])
                                 for i in range(1, self.n_conv)])
        
        # pooling layers
        self.pool_layers = nn.ModuleList([nn.MaxPool2d(self.p_kernels[0],
                                                       stride=self.p_strides[0],
                                                       padding=self.paddings[0])])
        
        self.pool_layers.extend([nn.MaxPool2d(self.p_kernels[i],
                                              stride=self.p_strides[i],
                                              padding=self.paddings[i])
                                 for i in range(1, self.n_pool)])
    
        # fully connected layers (CNN)
        self.fc_layers = nn.ModuleList([nn.Linear(self.flat_dim, self.fc_dims[0])])
        self.fc_layers.extend([nn.Linear(self.fc_dims[i-1], self.fc_dims[i]) for i in range(1, self.n_fc)])

        # fully connected layers (MLP)
        self.mlp_layers = nn.ModuleList([nn.Linear(self.select, self.mlp_dims[0])])
        self.mlp_layers.extend([nn.Linear(self.mlp_dims[i-1], self.mlp_dims[i]) for i in range(1, self.n_mlp)])                                              


    def cnn(self, X, get_activations=False):
        activations_cnn = []
        h = X
        for i in range(self.n_conv):
            h = self.conv_layers[i](h)
            h = F.relu(h)                                                                            
            activations_cnn.append(h)
            h = self.pool_layers[i](h)
            
        # flatten the activation before applying the fc layers
        h = h.reshape(1, -1)

        # forward pass through the fc layers
        for i in range(self.n_fc-1):
            h = self.fc_layers[i](h)
            h = F.relu(h)                                         
            activations_cnn.append(h)     

        h = self.fc_layers[self.n_fc-1](h)

        if get_activations:
            return h, activations_cnn
        else:
            return h

    def mlp(self, scores, get_activations=False): 
        activations_mlp = []
        h = scores
        for i in range(self.n_mlp-1):
            h = self.mlp_layers[i](h)
            h = F.relu(h)
                
        y = self.mlp_layers[self.n_mlp-1](h)

        if get_activations:
            return y, activations_mlp
        else:
            return y
                                                 
    def forward(self, tiles, aggregation):   
        if aggregation == 'mlp':
            htiles = torch.zeros((0)).to(self.device)

            for ii in range(len(tiles)):
                h = self.cnn(tiles[ii].unsqueeze(0))
                scores = softargmax(h, self.device)
                htiles = torch.cat((htiles, scores.unsqueeze(0)), dim=0)

            y = self.mlp(torch.transpose(htiles,0,1))

            return y
        
        else:
            y = torch.zeros((0,4)).to(self.device)

            for ii in range(len(tiles)):
                h = self.cnn(tiles[ii].unsqueeze(0))
                y = torch.cat((y, h), dim=0)

            return y
    
    def predict(self, X, aggregation): #Computes the probabilities of each class for each example in X. 
        logits = self.forward(X, aggregation)
        probs = F.softmax(logits, dim=-1)
        
        return probs


#----------------------------------------------- Tile selection --------------------------------------------------
def tile_selection_step(y_tiles, n_tiles=300):

    idx = torch.argsort(y_tiles, dim=0, descending=True)
    ten_pcent_idx = int(0.15*len(idx))

    if ten_pcent_idx < n_tiles:
        idx = idx[:n_tiles]
    
    else:
        idx = idx[:ten_pcent_idx]

    # Choose the best n_tiles indices from the indices list
    if len(idx) <= n_tiles: # Case 1: We have less or equal than 450 idx
        final_idx = idx
    
    elif len(idx) > n_tiles: # Case 2: We have more than 450 idx
        idx = idx.cpu()
        first_index = np.array([idx[0]])
        x = np.arange(1, len(idx)-1, (len(idx)-2)/(n_tiles-2))
        xp = np.arange(1, len(idx)-1)
        fp = np.arange(1, len(idx)-1)
        intermediate_indices = np.interp(x, xp, fp)
        intermediate_indices = [idx[int(a)] for a in intermediate_indices]
        last_index = np.array([idx[-1]])

        final_idx = np.concatenate((first_index, intermediate_indices, last_index))
        final_idx = np.array(final_idx, dtype='int')

    return final_idx


#---------------------------------------------- Train/Validation -------------------------------------------------
def train_model(device, filename, model, aggregation, loss_fn, optimizer, n_epochs, tiles_step, train_path, val_path='', 
                transform=None, SHUFFLE=False, BATCH_TILE=300, NUM_WORK=8):

    train_hist, val_hist, val_acc, val_F1 = [], [], [], []
    best_acc, best_f1 = 0., 0.

    train_files_ = np.array([f.split('.')[0] for f in os.listdir(train_path) if f.endswith('.pkl') ])
    
    if val_path is not '':
        val_files = np.array([d.split('.')[0] for d in os.listdir(val_path) if d.endswith('.pkl')])

    IMGS_BEFORE_STEP = tiles_step
    for epoch in range(n_epochs):
        start_time = time.time()
        print('\nEpoch', epoch+1)

        random.shuffle(train_files_)

        # Sort files as 010101...
        train_gt = np.array([int(f.split('_')[-2]) for f in train_files_])
        train_all = [np.argwhere(train_gt == 0), np.argwhere(train_gt == 1)]
        train_files = []

        for ll in range(max(len(train_all[0]), len(train_all[1]))):
            for kk in range(2):
                if ll >= len(train_all[kk]):
                    rand = np.random.choice(train_all[kk][:,0])
                    train_files.append(train_files_[rand])
                
                else:
                    train_files.append(train_files_[train_all[kk][ll, 0]])

        for j in range(len(train_files)):
            j_time = time.time()

            ff_set = dataset(os.path.join(train_path, train_files[j] + ''), train_files[j])
            ff_loader = DataLoader(ff_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)
            target = ff_set.target
            
            # Tile selection (M tiles with the highest scores after the CNN)
            with torch.no_grad():
                y_tiles = torch.zeros((0)).to(device)
                for i, (X, y) in enumerate(ff_loader):
                    for ii in range(len(X)):
                        tile = X[ii].unsqueeze(0).to(device, dtype=torch.float)  
                        score = model.cnn(tile)

                        if aggregation == 'mlp':       
                            y_tiles = torch.cat((y_tiles, softargmax(score, device)), dim=0)
                        
                        elif aggregation == 'median' or aggregation == 'mean': 
                            pred = torch.argmax(score).unsqueeze(0)
                            y_tiles = torch.cat((y_tiles, pred.to(dtype=torch.float)), dim=0)

                final_idx = tile_selection_step(y_tiles, n_tiles=BATCH_TILE)

            del ff_set, ff_loader, tile, score, y_tiles
            train_set = dataset(os.path.join(train_path, train_files[j] + ''), train_files[j], lst=final_idx)
            train_loader = DataLoader(train_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

            # Model training loop
            model.train()
            for k, (X, yy) in enumerate(train_loader):  
                X = X.to(device, dtype=torch.float)
                yscore = model(X, aggregation)

                if aggregation == 'mlp': 
                    y = target.unsqueeze(0).to(device)
                    ypred = torch.argmax(yscore, dim=-1) 
                    ypred = ypred.item()

                elif aggregation == 'median' or aggregation == 'mean':  
                    y = yy.to(device)
                    ypred = torch.argmax(yscore, dim=1)

                    if aggregation == 'median':
                        ypred, _ = torch.median(ypred.type(torch.float).cpu(), dim=-1)

                    elif aggregation == 'mean':
                        ypred = np.round(torch.mean(ypred.type(torch.float), dim=-1).cpu().detach().numpy())

                    if ypred == 0 or ypred == 1:
                        ypred = 0

                    elif ypred == 3 or ypred == 2:
                        ypred = 1
                
                loss = loss_fn(yscore, y)
                bloss = loss/IMGS_BEFORE_STEP
                bloss.backward()

            # Accumulate gradients before backpropagation
            if (j+1) % (IMGS_BEFORE_STEP) == 0:
                optimizer.step() 
                optimizer.zero_grad() 

            sys.stdout.write('\r.... Training: {:2}/{:2} | pred/gt: {}/{} | WSI loss: {:05.3f}'.format(
                                j+1, len(train_files), ypred, target, loss.item()))
            sys.stdout.flush()
        print()

        #compute validation loss to monitor the training progress (optional)
        with torch.no_grad():         
            model.eval()
            if val_path is not '': 

                targets, val_preds = [], []
                val_loss = 0.
                for j in range(len(val_files)):
                    ff_set = dataset(os.path.join(val_path, val_files[j] + ''), val_files[j])
                    ff_loader = DataLoader(ff_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)
                    target = ff_set.target
                    targets = np.append(targets, target)

                    y_tiles = torch.zeros((0,)).to(device)
                    for i, (X, _) in enumerate(ff_loader):
                        for ii in range(len(X)):
                            tile = X[ii].unsqueeze(0).to(device, dtype=torch.float)     
                            score = model.cnn(tile)

                            if aggregation == 'mlp':       
                                y_tiles = torch.cat((y_tiles, softargmax(score, device)), dim=0)
                        
                            else:
                                pred = torch.argmax(score).unsqueeze(0)
                                y_tiles = torch.cat((y_tiles, pred.to(dtype=torch.float)), dim=0)
                    
                    final_idx = tile_selection_step(y_tiles, n_tiles=BATCH_TILE)
                    
                    del ff_set, ff_loader, tile, score, y_tiles
                    val_set = dataset(os.path.join(val_path, val_files[j] + ''), val_files[j], lst=final_idx)
                    val_loader = DataLoader(val_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

                    for l, (X, yy) in enumerate(val_loader):
                        X = X.to(device, dtype=torch.float)
                        yscore = model(X, aggregation)

                        if aggregation == 'mlp': 
                            y = target.unsqueeze(0).to(device)
                            val_loss += loss_fn(yscore, y)
                            ypred = torch.argmax(model.predict(X, aggregation))
                            val_preds = np.append(val_preds, ypred.cpu())

                        else:
                            y = yy.to(device)
                            val_loss += loss_fn(yscore, y)
                            ypred = torch.argmax(yscore, dim=1)

                            if aggregation == 'median':
                                ypred, _ = torch.median(ypred.type(torch.float).cpu(), dim=-1)

                            elif aggregation == 'mean':
                                ypred = np.round(torch.mean(ypred.type(torch.float), dim=-1).cpu().detach().numpy())

                            if ypred == 0 or ypred == 1:
                                ypred = 0

                            elif ypred == 3 or ypred == 2:
                                ypred = 1

                            val_preds = np.append(val_preds, ypred)

                    sys.stdout.write('\r.... Validation: {:2}/{:2} | pred/gt: {}/{} | WSI loss: {:05.3f}'.format(
                                    j+1, len(val_files), ypred, target, loss_fn(yscore, y).item()))
                    sys.stdout.flush()
                print()

                val_loss /= j+1
                val_hist.append(val_loss.item())

                # Calculate metrics
                ACC, F1, precision, recall = get_metrics(val_preds, targets)           
                val_acc = np.append(val_acc, ACC)
                val_F1 = np.append(val_F1, F1)

                print('\r.... Validation loss: {:.3f} | ACC: {:.3f} | F1: {:.3f} | PRECISION: {:.3f} | RECALL: {:.3f}'.format(val_loss, ACC, F1, precision, recall))
                print('.... Elapsed time: {}'.format(timedelta(seconds=int(round(time.time() - start_time)))))
                
                # Save best model (with higher accuracy and F1)
                if ACC >= best_acc and F1 >= best_f1:

                    if not os.path.exists('./models/'):
                        os.mkdir('./models/')

                    elif not os.path.exists('./aux/'):
                        os.mkdir('./aux/')

                    model_file = './models/' + filename + '_' + aggregation + '.pth.tar'

                    cmatrix = metrics.confusion_matrix(targets, val_preds)
                    torch.save({'epoch': epoch + 1,
                                'best_ACC': ACC,
                                'best_F1': F1,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict()}, model_file)

                    f = open('./aux/' + filename + '_' + aggregation + '_metrics.pkl', 'wb')
                    pickle.dump([cmatrix, ACC, F1, precision, recall], f)
                    f.close()

                    print('........ Saving a new best model: {:.3f} --> {:.3f}'.format(best_acc, ACC))
                    best_acc = ACC
                    best_F1 = F1
                
                else:
                    print('........ Best F1: {:.3f} | Best ACC: {:.3f}'.format(best_f1, best_acc))

    print('\nTotal time:{} | Best ACC: {:.3f} | Best F1: {:.3f}'.format(timedelta(seconds=int(round(time.time() - start_time))), best_acc, best_f1))      
    return train_hist, val_hist, val_acc, val_F1



#----------------------------------------------------- Test ------------------------------------------------------
def test_model(device, model, aggregation, loss_fn, test_path, SHUFFLE=False, BATCH_TILE=300, NUM_WORK=8):

    test_files = np.array([f.split('.')[0] for f in os.listdir(test_path) if f.endswith('') ])

    with torch.no_grad():         
        model.eval()

        start_time = time.time()
        targets, test_preds = [], []
        test_loss = 0.
        for j in range(len(test_files)):
            ff_set = dataset(os.path.join(test_path, test_files[j] + ''), test_files[j])
            ff_loader = DataLoader(ff_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)
            target = ff_set.target
            targets = np.append(targets, target)

            # Tile selection (M tiles with the highest scores after the CNN)
            y_tiles = torch.zeros((0,)).to(device)
            for i, (X, _) in enumerate(ff_loader):
                for ii in range(len(X)):
                    tile = X[ii].unsqueeze(0).to(device, dtype=torch.float)     
                    score = model.cnn(tile)

                    if aggregation == 'mlp':       
                        y_tiles = torch.cat((y_tiles, softargmax(score, device)), dim=0)
                
                    else:
                        pred = torch.argmax(score).unsqueeze(0)
                        y_tiles = torch.cat((y_tiles, pred.to(dtype=torch.float)), dim=0)
            
            final_idx = tile_selection_step(y_tiles, n_tiles=BATCH_TILE)
            
            del ff_set, ff_loader, tile, score, y_tiles
            test_set = dataset(os.path.join(test_path, test_files[j] + ''), test_files[j], lst=final_idx)
            test_loader = DataLoader(test_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

            # Model inference
            for l, (X, _) in enumerate(test_loader):
                X = X.to(device, dtype=torch.float)
                yscore = model(X, aggregation)
                
                if aggregation == 'mlp': 
                    ypred = torch.argmax(model.predict(X, aggregation))
                    test_preds = np.append(test_preds, ypred.cpu())

                else:
                    ypred = torch.argmax(yscore, dim=1)
                    # ypred = torch.argmax(yscore, dim=1).to(dtype=torch.float)

                    if aggregation == 'median':
                        # ypred, _ = torch.median(ypred.unsqueeze(0), dim=-1)
                        ypred, _ = torch.median(ypred.type(torch.float).cpu(), dim=-1)

                    elif aggregation == 'mean':
                        ypred = np.round(torch.mean(ypred.type(torch.float), dim=-1).cpu().detach().numpy())

                    if ypred == 0 or ypred == 1:
                        ypred = 0

                    elif ypred == 3 or ypred == 2:
                        ypred = 1

                    test_preds = np.append(test_preds, ypred)

            sys.stdout.write('\r.... Test: {:2}/{:2} slides | pred/gt: {}/{}'.format(j+1, len(test_files), ypred, target))
            sys.stdout.flush()
        print()

        # Calculate metrics
        ACC, F1, precision, recall = get_metrics(test_preds, targets)           

        print('\r.... ACC: {:.3f} | F1: {:.3f} | PRECISION: {:.3f} | RECALL: {:.3f}'.format(ACC, F1, precision, recall))
        print('.... Elapsed time: {}'.format(timedelta(seconds=int(round(time.time() - start_time)))))

    return ACC, F1, precision, recall


#-------------------------------------------------- Inference ----------------------------------------------------
def inference(device, model, aggregation, data_path, SHUFFLE=False, BATCH_TILE=300, NUM_WORK=8):

    data_files = np.array([f.split('.')[0] for f in os.listdir(data_path) if f.endswith('') ])

    with torch.no_grad():        
        model.eval()

        preds = []
        for j in range(len(data_files)):
            start_inference = time.time()

            data_set = dataset(os.path.join(data_path, data_files[j] + ''), data_files[j], inference=True)
            data_loader = DataLoader(data_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

            # Tile selection (M tiles with the highest scores after the CNN)
            y_tiles = torch.zeros((0,)).to(device)
            for i, (X) in enumerate(data_loader):
                for ii in range(len(X)):
                    tile = X[ii].unsqueeze(0).to(device, dtype=torch.float)     
                    score = model.cnn(tile)

                    if aggregation == 'mlp':       
                        y_tiles = torch.cat((y_tiles, softargmax(score, device)), dim=0)
                
                    else:
                        pred = torch.argmax(score).unsqueeze(0)
                        y_tiles = torch.cat((y_tiles, pred.to(dtype=torch.float)), dim=0)
            
            final_idx = tile_selection_step(y_tiles, n_tiles=BATCH_TILE)
            
            del data_set, data_loader, tile, score, y_tiles
            data_set = dataset(os.path.join(data_path, val_files[j] + ''), val_files[j], lst=final_idx, inference=True)
            data_loader = DataLoader(data_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

            # Model inference
            for l, (X) in enumerate(data_loader):
                X = X.to(device, dtype=torch.float)
                yscore = model(X, aggregation)

                if aggregation == 'mlp': 
                    ypred = torch.argmax(model.predict(X, aggregation))
                    preds = np.append(preds, ypred.cpu())

                else:
                    ypred = torch.argmax(yscore, dim=1)

                    if aggregation == 'median':
                        ypred, _ = torch.median(ypred.type(torch.float).cpu(), dim=-1)

                    elif aggregation == 'mean':
                        ypred = np.round(torch.mean(ypred.type(torch.float), dim=-1).cpu().detach().numpy())

                    if ypred == 0 or ypred == 1:
                        ypred = 0

                    elif ypred == 3 or ypred == 2:
                        ypred = 1

                    preds = np.append(preds, ypred)

            sys.stdout.write('\r.... Inference: {:2}/{:2} | prediction: {}'.format(j+1, len(data_files), ypred))
            sys.stdout.flush()
        print()       
        print('.... Elapsed time: {}'.format(timedelta(seconds=int(round(time.time() - start_inference)))))

    return preds


#--------------------------------------------------- Metrics -----------------------------------------------------
def get_metrics(preds, targets):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ACC = metrics.accuracy_score(targets, preds)
        F1 = metrics.f1_score(targets, preds)#, zero_division=0) 
        precision = metrics.precision_score(targets, preds)#, zero_division='0')
        recall = metrics.recall_score(targets, preds)#, zero_division='0')

    return ACC, F1, precision, recall
