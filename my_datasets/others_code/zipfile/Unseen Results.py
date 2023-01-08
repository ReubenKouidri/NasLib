#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:27:34 2021

@author: josephcullen
"""
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score
from Data_generator_unseen import DataGenerator

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': True}

labels = np.load('OHE_unseen1.npy')

for i in range(1):
    foldnum = i 
    # Loaded 3rd one since it's closest to the average accuracy score 
    model = keras.models.load_model('final_2D_AFIBN_chatt.h5')
    
    
    #samples = [] 
    #for i in range(4000):
        #samples.append(i)   
    #unseen_datagenerator = DataGenerator(samples,labels,**params)
    #unseen_predictions = model.predict_generator(unseen_datagenerator, verbose = True)
    
    data = [] 
    for i in range(4000):
        temp = np.load('Unseen1/sample{}.npy'.format(i))
        data.append(temp)
        
    data = np.array(data)
    unseen_predictions = model.predict(data,batch_size = None,verbose = True)
    
    
    maxes = np.argmax(unseen_predictions, axis = 1)
    for c,i in enumerate(maxes):
        unseen_predictions[c] = np.zeros((2,))
        unseen_predictions[c][i] = 1
        
    unseen_cat = []
    for i in labels:
        unseen_cat.append(np.argmax(i))
    unseen_predcat = []
    for i in unseen_predictions:
        unseen_predcat.append(np.argmax(i))
    
    conf_matrix_unseen = confusion_matrix(unseen_cat,unseen_predcat)
    plt.figure()
    plot_confusion_matrix(conf_matrix_unseen, classes = ['AFIB','N'])
    plt.savefig('Conf Matrix 3 CWT Beat UNSEEN no norm FINAL.pdf') 
    acc = accuracy_score(labels, unseen_predictions)
    print('Accuracy = ', acc)
    f1 = f1_score(labels, unseen_predictions, average = None)
    f1_tot = np.zeros((2,))
    f1_tot = f1_tot + f1
    print(f1_tot)

