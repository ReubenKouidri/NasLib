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
from Arrhythmia_generator_1D_binary import DataGenerator
#from Arrhythmia_generator_aug_less_classes import DataGenerator_aug
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras import layers, Input, Model, optimizers
from tensorflow.keras.layers import Multiply, Add
from tensorflow.keras.activations import sigmoid
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
    if epoch < 21:
        lr = initial_lrate
    else:
        lr = lower_lrate
    print(lr)
    return lr


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
# In[ ]:
import os
print(os.getcwd() )
# Load in  file indices and labels
with open('1DCNN_binary_3_Beat_Train_Labels_OHE.pkl', 'rb') as f:
    file_labels = pickle.load(f)
file_indices = []
for i in range(18000):
    file_indices.append(i)
# In[ ]:

# Parameters
params = {'dim': (2800),
          'batch_size': 25,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (2800),
          'batch_size': 25,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': False}

params_test = {'dim': (2800),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 2,
          'shuffle': False}

# In[ ]:
f1_tot = np.zeros((2,))
# Splitting the data into test and train sets 
X_final = file_indices
Y_final = file_labels
# X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.20)


num_classes = 2

# Model architecture
input_img = Input(shape = (2800,2), name = '1Dseries')

# CONV BLOCK 1
conv1 = Conv1D(32, kernel_size = 3, activation='relu')(input_img)
bn = BatchNormalization()(conv1)
maxpool1 = MaxPooling1D(2)(bn)
conv2 = Conv1D(32, kernel_size = 3, activation='relu')(maxpool1)
maxpool2 = MaxPooling1D(2)(conv2)
#dropout1 = Dropout(0.5)(maxpool1)

"""# CHANNEL ATTENTION - dropout1 provides the input
# Use both global max and average pooling
maxpool1 = GlobalMaxPooling1D()(maxpool1)
avpool1 = GlobalAveragePooling1D()(maxpool1)
# Define two dense layers to function as a MLP with one hidden layer
CA1_dense1 = Dense(8, activation='relu')
CA1_dense2 = Dense(32, activation='relu')
# Feed both pooling into MLP
max_out1 = CA1_dense2(CA1_dense1(maxpool1))
av_out1 = CA1_dense2(CA1_dense1(avpool1))
# Sum and sigmoid
layer_sum1 = Add()([max_out1, av_out1])
CA1_output = sigmoid(layer_sum1)
# Now perform element wise multiplication with input feature map
Chat_out1 = Multiply()([maxpool1, CA1_output])"""

# CONV BLOCK 2
conv3 = Conv1D(64, kernel_size = 5, activation='relu')(maxpool2)
maxpool3 = MaxPooling1D(2)(conv3)
conv4 = Conv1D(64, kernel_size = 5, activation='relu')(maxpool3)
maxpool4 = MaxPooling1D(2)(conv4)
#dropout2 = Dropout(0.5)(maxpool2)

# CONV BLOCK 3
conv5 = Conv1D(128, kernel_size = 7, activation='relu')(maxpool4)
maxpool5 = MaxPooling1D(2)(conv5)
conv6 = Conv1D(128, kernel_size = 7, activation='relu')(maxpool5)
maxpool6 = MaxPooling1D(2)(conv6)

"""# CHANNEL ATTENTION - dropout2 provides the input
# Use both global max and average pooling
maxpool2 = GlobalMaxPooling1D()(dropout2)
avpool2 = GlobalAveragePooling1D()(dropout2)
# FDefine two dense layers to function as a MLP with one hidden layer
CA2_dense1 = Dense(4, activation='relu')
CA2_dense2 = Dense(32, activation='relu')
# Feed both pooling into MLP
max_out2 = CA2_dense2(CA2_dense1(maxpool2))
av_out2 = CA2_dense2(CA2_dense1(avpool2))
# Sum and sigmoid
layer_sum2 = Add()([max_out2, av_out2])
CA2_output = sigmoid(layer_sum2)
# Now perform element wise multiplication with input feature map
Chat_out2 = Multiply()([dropout2, CA2_output])"""
flatten1 = Flatten()(maxpool6)
dropout2 = Dropout(0.5)(flatten1)
dense1 = Dense(64, activation='relu')(dropout2)
dropout3 = Dropout(0.5)(dense1)
dense2 = Dense(20, activation='relu')(dropout3)
dense3 = Dense(num_classes, activation='softmax')(dense2)

model = Model(input_img, dense3, name = 'old_model')    
sgd = optimizers.SGD(lr = 0.001, momentum = 0.9)
model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])
# learning rate schedule
lrate = LearningRateScheduler(step_decay)
callback_list = [lrate]
train_generator = DataGenerator(X_final, Y_final, **params)
print(model.summary())
print('=============================================')  
history = model.fit_generator(generator=train_generator,
                          epochs=50, verbose=1)
""", callbacks = callback_list"""
scores_generator = DataGenerator(X_final, Y_final, **params_test)
flat_predictions = model.predict_generator(scores_generator, verbose = True)
# Take largest prediction as the actual prediction
maxes = np.argmax(flat_predictions, axis = 1)
for c,i in enumerate(maxes):
    flat_predictions[c] = np.zeros((2,))
    flat_predictions[c][i] = 1
    
print('=============================================')
print('ACCURACY: ')
acc = accuracy_score(Y_final, flat_predictions)
print(acc)
# Save model
model.save('final_model_save.h5')
with open('final_history.pkl', 'wb') as f:
    pickle.dump(history.history,f)


print('folds complege')
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

