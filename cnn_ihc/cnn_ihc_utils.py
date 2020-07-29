#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:36:31 2019
@author: sipo, jtrp & tfg
INESCTEC
"""

import os
import sys
import math
import time
import random
import pickle
import numpy as np
from PIL import Image
from datetime import timedelta
import sklearn.metrics as metrics

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

#-------------------------------------------- Dataset ----------------------------------------------
class dataset():
    def __init__(self, path, files, transform=None, train=False):
       
        self.path = path
        
        patches = []
        targets = []
        for ii in range(len(files)):
            pik = open(os.path.join(path,files[ii]), 'rb')
            tiles, _ , label = pickle.load(pik)
            pik.close()
            
            for jj in range(len(tiles)):
                patches.append(tiles[jj])

            b = [label]*len(tiles)
            targets = np.concatenate((targets,b))

        self.patches = patches
        self.targets = targets
        self.transform = transforms.ToTensor()
       
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        a = Image.fromarray(self.patches[idx])
        tile = np.array(a.resize((256,256), Image.ANTIALIAS))
        tile = np.reshape(tile, (256,256,1))

        tile = self.transform(tile) 
        label = torch.tensor(self.targets[idx])
        label = label.long()

        return tile, label


#------------------------------------------- CNN model ---------------------------------------------
def get_flat_dim(input_dim, n_conv, conv_filters, kernel_sizes, p_kernel, strides, p_strides, paddings):
    _, H, W = input_dim
    for i in range(n_conv):
        H = int(math.floor((H + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        H = int(math.floor((H - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
        W = int(math.floor((W + 2*paddings[i] - kernel_sizes[i])/(1.*strides[i]) + 1))
        W = int(math.floor((W - p_kernel[i])/(1.*p_strides[i]) + 1)) # 0 padding in pooling
    
    flat_dim = H * W * conv_filters[-1]
    
    return flat_dim
    
    
class CNN_IHC(nn.Module):
    def __init__(self, n_conv, n_pool, n_fc, conv_filters, kernel_sizes, p_kernels, strides, p_strides,
                 paddings, fc_dims, in_channels, flat_dim):

        super(CNN_IHC, self).__init__()
        
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
        self.pool_layers = nn.ModuleList([nn.MaxPool2d(self.p_kernels[0],
                                                       stride=self.p_strides[0],
                                                       padding=self.paddings[0])])
        
        self.pool_layers.extend([nn.MaxPool2d(self.p_kernels[i],
                                              stride=self.p_strides[i],
                                              padding=self.paddings[i])
                                 for i in range(1, self.n_pool)])
    
        # fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.flat_dim, self.fc_dims[0])])
        self.fc_layers.extend([nn.Linear(self.fc_dims[i-1], self.fc_dims[i]) 
                                for i in range(1, self.n_fc)])                                         
                                                                                              
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

        y = self.fc_layers[self.n_fc-1](h)
        
        if get_activations:
            return y, activations
        else:
            return y
    
    def predict(self, X): #Computes the probabilities of each class for each example in X. 
        
        logits = self.forward(X)
        probs = F.softmax(logits, dim=1)
        
        return probs

#------------------------------------------- CNN train ---------------------------------------------
def train_model(device, mfilename, model, loss_fn, optimizer, n_epochs, train_path, val_path='', transform=None, SHUFFLE=True, BATCH_TILE=128, NUM_WORK=8):

    train_hist, valid_hist, train_acc, valid_acc = [], [], [], []
    best_acc = 0.

    train_files = np.array([f.split('.')[0] for f in os.listdir(train_path)])

    if val_path is not '':
        val_files = np.array([d.split('.')[0] for d in os.listdir(val_path)])

    train_set = dataset(train_path, train_files)
    train_loader = DataLoader(train_set, batch_size=BATCH_TILE, shuffle=True, num_workers=NUM_WORK)

    val_set = dataset(val_path, val_files)
    val_loader = DataLoader(val_set, batch_size=BATCH_TILE, shuffle=SHUFFLE, num_workers=NUM_WORK)

    for epoch in range(n_epochs):
        start_time = time.time()
        print('\nEpoch', epoch+1)

        for i, (X, y) in enumerate(train_loader):             
            X = X.to(device)        
            y = y.to(device)              

            ypred = model(X)
            loss = loss_fn(ypred, y) 
            optimizer.zero_grad()     
            loss.backward()           
            optimizer.step()          

            sys.stdout.write("\r" + '.... Training: {}/{} mini-batch'.format(i+1, len(train_loader)))
            sys.stdout.flush()  
        print()

        #compute train & validation loss to monitor the training progress (optional)
        with torch.no_grad():          
            model.eval()

            tloss, n_correct, N = 0., 0., 0.
            for m, (X, y) in enumerate(train_loader):                        
                X = X.to(device) 
                y = y.to(device)

                scores = model(X)       
                tloss += loss_fn(scores, y)  

                ypred = torch.argmax(scores, dim=1)
                n_correct += torch.sum(1.*(ypred == y)).item()
                N += len(y)
            
            tloss /= m+1
            train_hist.append(tloss.item())
            acc_train = n_correct / N
            train_acc.append(float("{:.2f}".format(acc_train)))
            print('.... Training loss: {:.3f} | training ACC: {:.3f}'.format(tloss, acc_train))

            if val_path is not '': 

                val_loss, n_correct, N = 0., 0., 0.
                preds, gt = [], []

                for i, (X, y) in enumerate(val_loader):
                    gt = np.append(gt, y)

                    X = X.to(device)
                    y = y.to(device)
                
                    scores = model(X)         
                    val_loss += loss_fn(scores, y)  

                    ypred = torch.argmax(scores, dim=1)
                    preds = np.append(preds, ypred.cpu())
                    n_correct += torch.sum(1.*(ypred == y)).item()
                    N += len(y)

                val_loss /= i + 1
                valid_hist.append(val_loss.item())
                acc_val = n_correct / N
                valid_acc.append(float("{:.2f}".format(acc_val)))
                print('.... Validation loss: {:.3f} | validation ACC: {:.3f}'.format(val_loss, acc_val))


            if acc_val > best_acc:
                if not os.path.exists('./models/'):
                    os.mkdir('./models/')

                elif not os.path.exists('./aux/'):
                    os.mkdir('./aux/')

                model_file = './models/' + mfilename + '_IHC.pth.tar'

                confusion_matrix = metrics.confusion_matrix(gt, preds)
                
                torch.save({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_acc': acc_val,
                                'optimizer' : optimizer.state_dict()}, model_file)

                print('....... Saving a new best model: {:.3f} --> {:.3f}'.format(best_acc, acc_val))
                best_acc = acc_val
                best_epoch = epoch + 1

                f = open('./aux/' + mfilename + '_IHC_cmatrix.pkl', 'wb')
                pickle.dump([confusion_matrix], f)
                f.close()

        f = open('./aux/' + mfilename + '_IHC.pkl', 'wb')
        pickle.dump([train_hist, valid_hist, train_acc, valid_acc, best_acc], f)
        f.close()

    print('Best validation accuracy: {:.3f} | epoch {}'.format(best_acc, best_epoch))
    print('Total time:',timedelta(seconds=int(round(time.time() - start_time))))       
    return train_hist, valid_hist, valid_acc


#-------------------------------------------- Metrics ----------------------------------------------
def get_metrics(scores, targets):
    
    TN, FP, FN, TP = metrics.confusion_matrix(targets, scores).ravel()
    F1 = metrics.fbeta_score(targets, scores, beta=1)  

    return  TN, FP, FN, TP, F1
