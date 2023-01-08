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
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import layers, Input, Model, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
#from random_eraser import get_random_eraser
from tensorflow.python.keras.utils.data_utils import Sequence
print(tf.__version__)


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
with open('CNN_3_Beat_Train_Labels_OHE.pkl', 'rb') as f:
    file_labels = pickle.load(f)
file_indices = []
for i in range(7000):
    file_indices.append(i)
# In[ ]:

# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 7,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 7,
          'n_channels': 2,
          'shuffle': False}

params_test = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 7,
          'n_channels': 2,
          'shuffle': False}

# In[ ]:
f1_tot = np.zeros((7,))
# Splitting the data into test and train sets 
X_final = file_indices
Y_final = file_labels
X_train, X_val, Y_train, Y_val = train_test_split(X_final, Y_final, test_size=0.20, stratify=Y_final)

lr_values = [0.1,0.01,0.001,0.0001,0.00001]
model_accuracies = []
for lr in lr_values:

    print(lr)
    num_classes = 7
    
    # Model architecture
    input_img = Input(shape = (128,128,2), name = 'CWTimage')
    # CONV BLOCK 1
    conv1 = Conv2D(32, kernel_size = 10, activation='relu')(input_img)
    conv2 = Conv2D(32, kernel_size = 10, activation='relu')(conv1)
    maxpool1 = MaxPooling2D((2, 2))(conv2)
    dropout1 = Dropout(0.5)(maxpool1)
    #CONV BLOCK 2
    conv3 = Conv2D(32, kernel_size = 8, activation='relu')(dropout1)
    conv4 = Conv2D(16, kernel_size = 4, activation='relu')(conv3)
    maxpool2 = MaxPooling2D((2, 2))(conv4)
    dropout2 = Dropout(0.5)(maxpool2)
    # Flatten and Dense layers
    flatten1 = Flatten()(dropout2)
    dense1 = Dense(128, activation='relu')(flatten1)
    dropout3 = Dropout(0.5)(dense1)
    dense2 = Dense(num_classes, activation='softmax')(dropout3)
    
    model = Model(input_img, dense2, name = 'old_model')
    sgd = optimizers.SGD(learning_rate=lr, momentum=0.9, decay=1e-6, nesterov = True)
    model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])

    train_generator = DataGenerator(X_train, Y_train, **params)
    val_generator = DataGenerator(X_val, Y_val, **params_val)
    
    print('=============================================')
    print('TRAINING FOR LEARNING RATE = {}: '.format(lr))
    
    history = model.fit_generator(generator=train_generator,
                              epochs=50, validation_data=val_generator,
                              verbose=True)
    
    scores_generator = DataGenerator(X_val, Y_val, **params_test)
    flat_predictions = model.predict_generator(scores_generator, verbose = True)
    # Take largest prediction as the actual prediction
    maxes = np.argmax(flat_predictions, axis = 1)
    for c,i in enumerate(maxes):
        flat_predictions[c] = np.zeros((7,))
        flat_predictions[c][i] = 1
        
    print('=============================================')
    print('ACCURACY: ')
    acc = accuracy_score(Y_val, flat_predictions)
    print(acc)
    model_accuracies.append(acc)
    # Save model
    model.save('model_saves/lr_{}_model_save.h5'.format(lr))
    with open('history_saves/lr_{}_history.pkl'.format(lr), 'wb') as f:
        pickle.dump(history.history,f)
    print('*** LEARNING RATE =  ', lr, ' MODEL SAVED ***')

# save accuracies to a file
file = open('accuracy.txt', 'w')
for i in range(len(lr_values)):    
    file.write('Learning Rate = {} val. accuracy: {}\n'.format(lr_values[i], model_accuracies[i]))
file.close()


print('trains complege')
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
