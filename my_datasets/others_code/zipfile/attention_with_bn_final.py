#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.keras
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append('P:/')
from Arrhythmia_generator import DataGenerator
#from Arrhythmia_generator_aug_less_classes import DataGenerator_aug
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import layers, Input, Model, optimizers
from tensorflow.keras.layers import Multiply, Add
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
#from random_eraser import get_random_eraser
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.callbacks import LearningRateScheduler
import math
 
# set learning rate schedule
def step_decay(epoch):
    # set schedule value
    initial_lrate = 0.001
    lower_lrate = 0.0001
    if epoch < 20:
        lr = initial_lrate
    else:
        lr = lower_lrate
    print(lr)
    return lr


# In[ ]:
import os
print(os.getcwd() )
# Load in  file indices and labels
# with open('CNN_3_Beat_Train_Labels_OHE.pkl', 'rb') as f:
#     file_labels = pickle.load(f)
file_labels = np.load('OHE_trainval.npy')
file_indices = []
for i in range(18000):
    file_indices.append(i)
# In[ ]:

# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': False}

params_test = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': False}

# In[ ]:
f1_tot = np.zeros((2,))
# Splitting the data into test and train sets 
X_final = file_indices
Y_final = file_labels

# Turn Y back from OHE bc Kfold doesn't work with OHE
Y_labels = []
for i in Y_final:
    Y_labels.append(np.argmax(i))
X_final = np.array(X_final)
Y_labels = np.array(Y_labels)
Y_train_OHE = to_categorical(Y_labels)
print('ur da')

# In[]
num_classes = 2

# Model architecture
input_img = Input(shape = (128,128,2), name = 'CWTimage')

# CONV BLOCK 1
conv1 = Conv2D(32, kernel_size = 10, activation='relu')(input_img)
bn1 = BatchNormalization()(conv1)
conv2 = Conv2D(32, kernel_size = 10, activation='relu')(bn1)
bn2 = BatchNormalization()(conv2)
maxpool1 = MaxPooling2D((2, 2))(bn2)

# CHANNEL ATTENTION - dropout1 provides the input
# Use both global max and average pooling
CAmaxpool1 = GlobalMaxPooling2D()(maxpool1)
CAavpool1 = GlobalAveragePooling2D()(maxpool1)
# FDefine two dense layers to function as a MLP with one hidden layer
CA1_dense1 = Dense(8, activation='relu')
CA1_dense2 = Dense(32, activation='relu')
# Feed both pooling into MLP
max_out1 = CA1_dense2(CA1_dense1(CAmaxpool1))
av_out1 = CA1_dense2(CA1_dense1(CAavpool1))
# Sum and sigmoid
layer_sum1 = Add()([max_out1, av_out1])
CA1_output = sigmoid(layer_sum1)
# Now perform element wise multiplication with input feature map
Chat_out1 = Multiply()([maxpool1, CA1_output])

# CONV BLOCK 2
conv3 = Conv2D(32, kernel_size = 8, activation='relu')(Chat_out1)
bn3 = BatchNormalization()(conv3)
conv4 = Conv2D(32, kernel_size = 4, activation='relu')(bn3)
bn4 = BatchNormalization()(conv4)
maxpool2 = MaxPooling2D((2, 2))(bn4)

# CHANNEL ATTENTION - dropout2 provides the input
# Use both global max and average pooling
CAmaxpool2 = GlobalMaxPooling2D()(maxpool2)
CAavpool2 = GlobalAveragePooling2D()(maxpool2)
# FDefine two dense layers to function as a MLP with one hidden layer
CA2_dense1 = Dense(8, activation='relu')
CA2_dense2 = Dense(32, activation='relu')
# Feed both pooling into MLP
max_out2 = CA2_dense2(CA2_dense1(CAmaxpool2))
av_out2 = CA2_dense2(CA2_dense1(CAavpool2))
# Sum and sigmoid
layer_sum2 = Add()([max_out2, av_out2])
CA2_output = sigmoid(layer_sum2)
# Now perform element wise multiplication with input feature map
Chat_out2 = Multiply()([maxpool2, CA2_output])

# Flatten and Dense layers
flatten1 = Flatten()(Chat_out2)
dense1 = Dense(100, activation='relu', kernel_regularizer=l2(0.0001))(flatten1)
dropout3 = Dropout(0.5)(dense1)
dense2 = Dense(num_classes, activation='softmax')(dropout3)

model = Model(input_img, dense2, name = 'old_model_with_attention')    
sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov = True)
model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])
# learning rate schedule
lrate = LearningRateScheduler(step_decay)
callback_list = [lrate]
train_generator = DataGenerator(X_final, Y_train_OHE, **params)

history = model.fit_generator(generator=train_generator,epochs=50, verbose=1)
# Save model
model.save('final_2D_AFIBN_chatt.h5')
with open('final_history.pkl', 'wb') as f:
    pickle.dump(history.history,f)
    
# In[ ]:    
"""
# Evaluate the initial model on the test data
print('=============================================')
print('TESTING UNTRAINED NETWORK: ')
result = model.evaluate(val_generator, verbose = True)
print(result)

print('UNTRAINED PREDICT: ')
predictions = model.predict(val_generator)
print(predictions[0])
print('Actual:')
print(Y_test[0])

print('=============================================')
print('TRAINING: ')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint/saved-model-{epoch:02d}-{val_loss:.2f}.h5',
    monitor='val_loss',
    mode='min',
    verbose=True,
    save_best_only=True)

history = model.fit_generator(generator=train_generator,
                              epochs=30, validation_data=val_generator,
                              callbacks=[model_checkpoint_callback],
                              verbose=True)

print('TESTING TRAINED NETWORK: ')
result = model.evaluate(val_generator)
print('Results:')
print(result)

print('TRAINED PREDICT')
predictions = model.predict(val_generator)
print(predictions[0])
print('Actual:')
print(Y_test[0])

# Save model
model.save('HengguiCNN/CNN_less_classes_less_norm.h5')
with open('HengguiCNN/History_CNN_less_classes_less_norm.pkl', 'wb') as f:
    pickle.dump(history.history,f)
print('*** FOLD ', fold_num, ' MODEL SAVED***')
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'], 'cornflowerblue')
plt.plot(history.history['val_categorical_accuracy'], 'lightcoral')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig('Accuracy plot.pdf', dpi = 1200)
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'], 'cornflowerblue')
plt.plot(history.history['val_loss'], 'lightcoral')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()
plt.savefig('Loss plot.pdf', dpi = 1200)

print('=============================================')
print('EVALUATING SCORES:')
scores_generator = DataGenerator(X_test, Y_test, **params_test)
flat_predictions = model.predict_generator(scores_generator,verbose = True)
print('PREDICTIONS:')
print(flat_predictions)
flat_labels = np.array(Y_test)

print('First 10 labels and predictions:')
for i in range(10):
    print(i,') Actual:', Y_test[i])
    print(i,') Predicted:', flat_predictions[i])

# Take largest prediction as the actual prediction
maxes = np.argmax(flat_predictions, axis = 1)
for c,i in enumerate(maxes):
    flat_predictions[c] = np.zeros((7,))
    flat_predictions[c][i] = 1
    
#print('Flat predictions again: ', flat_predictions)
#print(flat_predictions.shape)
#print(flat_labels.shape)

cat_true = []
for i in flat_labels:
    cat_true.append(np.argmax(i))
cat_predictions = []
for i in flat_predictions:
    cat_predictions.append(np.argmax(i))
# calculate confusion matrix
conf_matrix = confusion_matrix(cat_true, cat_predictions)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes = ['N','AFIB','P','B','SBR','AFL', 'T'],
                      title='Confusion matrix, no normalisation')
plt.savefig('Conf Matrix no norm.pdf')          
plt.figure()
plot_confusion_matrix(conf_matrix, classes = ['N','AFIB','P','B','SBR','AFL', 'T'], normalize = True,
                      title='Confusion matrix, normalised')
plt.savefig('Conf Matrix norm.pdf')   
        
print('CONFUSION MATRIX: ')
print(conf_matrix)
print('=============================================')
print('F1 SCORES: ')
print(f1_score(flat_labels, flat_predictions, average = None))
f1 = f1_score(flat_labels, flat_predictions, average = None)
f1_tot = f1_tot + f1
print(f1_tot)
print(f1.shape)

print('ACCURACY: ')
acc = accuracy_score(flat_labels, flat_predictions)
print(acc)
# save accuracy to a file
file = open('accuracy.txt', 'w')
file.write('accuracy = {}'.format(acc))
file.close()
print('=============================================')"""
