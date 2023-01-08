#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:38:09 2021

@author: josephcullen
"""
# import packages
from scipy import signal
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pywt
import PIL
from PIL import Image
from skimage.transform import resize
import math
from pywt._doc_utils import boundary_mode_subplot

def CWT(beat ,widths=np.arange(1,128)):
    # This performs a CWT of every channel of the segment, then adds and axis and stacks them
    # First find the number of channels in the segment
    num_leads = beat.shape[-1] 
    pad = 350
    # Before performing the CWT, the sides of the array need to be padded:
    padded_1 = np.pad(beat[:,0],(pad,pad),'edge')
    padded_2 = np.pad(beat[:,1],(pad,pad),'edge')
    beat = np.stack((padded_1,padded_2),axis=-1)
    
    # Now loop over every channel and perform the CWT:
    for i in range(num_leads):
        if i == 0:
            cwtmatr, freqs = pywt.cwt(beat[:,i], widths, 'cmor1.5-1')
            cwtha = abs(cwtmatr)
            
            # Need to cut off the ends of the CWT that were added to counter the edge effects
            cwtha = cwtha[:,pad:-pad]
            # Need to resize it to the correct shape
            resized1 = resize(cwtha, (128,128))
            
            # Normalise all the values 
            mean = np.mean(resized1)
            resized1 -= mean
            std = np.std(resized1)
            resized1 /= std
            
            resized1 = np.array(resized1, dtype=np.float32)
            sample = resized1[:,:,np.newaxis]
        else:
            cwtmatr, freqs = pywt.cwt(beat[:,i], widths, 'cmor1.5-1')
            cwtha = abs(cwtmatr)
            cwtha = cwtha[:,pad:-pad]
            resized1 = resize(cwtha, (128,128))
            
            # Normalise all the values 
            mean = np.mean(resized1)
            resized1 -= mean
            std = np.std(resized1)
            resized1 /= std

            resized1 = np.array(resized1, dtype=np.float32)
            resized1 = resized1[:,:,np.newaxis]
            sample = np.concatenate((sample,resized1),axis=-1)
        
    return sample

# This is a loop to perform a CWT of every sample
num_of_samples = 18000
for i in range(num_of_samples):
    with open('Train Data1/sample{}.pkl'.format(i), 'rb') as f:
        segment = np.load(f, allow_pickle = True)
    CWTed = CWT(segment)
    np.save('Train Data1/sample{}.npy'.format(i),CWTed ,allow_pickle = True)
    if i%500 == 0:
        print(i)
print('Train complete')

# This is a loop to perform a CWT of every unseen sample
num_of_samples = 4000
for i in range(num_of_samples):
    with open('Unseen1/sample{}.pkl'.format(i), 'rb') as f:
        segment = np.load(f, allow_pickle = True)
    CWTed = CWT(segment)
    np.save('Unseen1/sample{}.npy'.format(i),CWTed ,allow_pickle = True)
    if i%200 == 0:
        print(i)
print('Unseen complete')