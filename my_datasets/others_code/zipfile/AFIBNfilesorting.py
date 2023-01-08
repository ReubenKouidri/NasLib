#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:52:31 2021

@author: josephcullen
"""
import pickle 
import numpy as np 



AFIB_path = "AFIB/"
N_path = "N/"
TrainVal_path = "Train Data1/"
OHE_path_train = "OHE_train1.npy"
OHE_path_unseen = "OHE_unseen1.npy"
unseen_path = "Unseen1/"

OHE_trainval_labels = [] 
unseencount = 0
trainvalcount = 0 
for i in range(9000):
    samp = 'seg_{}.pkl'.format(i)
    movesamp = pickle.load(open( AFIB_path+samp, "rb" ) )
    pickle.dump(movesamp,open( TrainVal_path+'sample{}.pkl'.format(trainvalcount), "wb" ))
    OHE = np.array([1,0])
    OHE_trainval_labels.append(OHE)
    trainvalcount += 1  
    
for i in range(9000):
    samp = 'seg_{}.pkl'.format(i)
    movesamp = pickle.load(open( N_path+samp, "rb" ) )
    pickle.dump(movesamp,open( TrainVal_path+'sample{}.pkl'.format(trainvalcount), "wb" ))
    OHE = np.array([0,1])
    OHE_trainval_labels.append(OHE)
    trainvalcount += 1 
OHE_trainval_labels = np.array(OHE_trainval_labels)
np.save(OHE_path_train, OHE_trainval_labels)

OHE_unseen = [] 
for i in range(2000):
    samp = 'seg_{}.pkl'.format(i+9000)
    movesamp = pickle.load(open( AFIB_path+samp, "rb" ) )
    pickle.dump(movesamp,open( unseen_path+'sample{}.pkl'.format(unseencount), "wb" ))
    OHE = np.array([1,0])
    OHE_unseen.append(OHE)
    unseencount += 1
    
for i in range(2000):
    samp = 'seg_{}.pkl'.format(i+9000)
    movesamp = pickle.load(open( N_path+samp, "rb" ) )
    pickle.dump(movesamp,open( unseen_path+'sample{}.pkl'.format(unseencount), "wb" ))
    OHE = np.array([0,1])
    OHE_unseen.append(OHE)
    unseencount += 1 

OHE_unseen = np.array(OHE_unseen)
np.save(OHE_path_unseen, OHE_unseen)