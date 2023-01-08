#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:06:36 2021

@author: josephcullen
"""
import collections
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from Arrhythmia_generator import DataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix

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


foldnum = 1
model = load_model('fold_model_saves/fold_{}_model_save.h5'.format(foldnum))
with open('fold_history_saves/fold_{}_history.pkl'.format(foldnum), 'rb') as f:
    history = pkl.load(f)
    
file_labels = np.load('OHE_trainval.npy')
file_indices = []
for i in range(18000):
    file_indices.append(i)
    
X = np.array(file_indices)
Y = file_labels
# In[ ]:

# Parameters
params = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': True}
 
# Test model on all training data  
train_generator = DataGenerator(X, Y, **params)

OHE_predictions = model.predict_generator(train_generator, verbose = True)
print(Y[0:10])
print(OHE_predictions[0:10])
maxes = np.argmax(OHE_predictions, axis = 1)
for c, i in enumerate(maxes):
    OHE_predictions[c] = np.zeros((2,))
    OHE_predictions[c][i] = 1
print(OHE_predictions[0:10])

acc = accuracy_score(Y, OHE_predictions)
print('ACCURACY: ', acc)

# Confusion Matrix
cat_true = []
for i in Y:
    cat_true.append(np.argmax(i))
cat_predictions = []
for i in OHE_predictions:
    cat_predictions.append(np.argmax(i))
conf_matrix = confusion_matrix(cat_true, cat_predictions)
print(collections.Counter(cat_true))
print(collections.Counter(cat_predictions))
print(conf_matrix)


# Test model on all test data
file_labels = np.load('OHE_unseen.npy')
file_indices = []
for i in range(4000):
    file_indices.append(i)
    
X = np.array(file_indices)
Y = file_labels
test_generator = DataGenerator(X, Y, **params)

OHE_predictions = model.predict_generator(test_generator, verbose = True)
print(Y[0:4000])
print(OHE_predictions[0:10])
maxes = np.argmax(OHE_predictions, axis = 1)
for c, i in enumerate(maxes):
    OHE_predictions[c] = np.zeros((2,))
    OHE_predictions[c][i] = 1
print(OHE_predictions[0:10])

acc = accuracy_score(Y, OHE_predictions)
print('ACCURACY: ', acc)

# Confusion Matrix
cat_true = []
for i in Y:
    cat_true.append(np.argmax(i))
cat_predictions = []
for i in OHE_predictions:
    cat_predictions.append(np.argmax(i))
conf_matrix = confusion_matrix(cat_true, cat_predictions)

import collections
print(collections.Counter(cat_true))
print(collections.Counter(cat_predictions))

print(conf_matrix)







