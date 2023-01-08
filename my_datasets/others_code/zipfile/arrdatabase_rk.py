import wfdb
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Any, Tuple
import os
import pandas as pd


names = [100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,
         124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,
         231,232,233,234]


os.chdir("/Users/reubenkouidri/Documents/Uni/Physics/Year_4/MPhys Project/code/Project/mitdb")


def extract_data(filename):
    """d_signal just means 'digital signal' - from conversion of analog to digital """
    print(f'{filename}')
    record = wfdb.rdrecord(f'{filename}')
    d_signal = record.adc()
    # This normalises the data so that the max value is 1
    # V_signal = (V_signal - V_signal.min())/(V_signal.max() - V_signal.min())
    return d_signal


def extract_labels(filename):
    """
    extract the labels and their locations (ie the peak locations) for each beat and
        puts them into a lists
    rdann reads in an annotation file and returns an Annotation object
    """

    ann = wfdb.rdann(f'{filename}', 'atr', return_label_elements=['symbol'])
    # These two lines return the symbol and the locations
    labels_symbol = ann.symbol
    locations = ann.sample
    return labels_symbol, locations


for i, name in enumerate(names):
    signal = extract_data(name)
    df = pd.DataFrame(signal)
    df.to_pickle(f'arrays/{i}.pkl')
    #df.from_dict()
    #print(df.head())
    #with open('arrays/{}_array.csv'.format(i), 'wb') as f:
    #    pickle.dump(signal, f)
    #labels_symbol, peaks = extract_labels(name)
    #save_data(labels_symbol, folder='labels', file_name=i)
    #save_data(peaks, folder='label_locations', file_name=i)

"""
for i, name in enumerate(names):
    labels_symbol, peaks = extract_labels(name)
    with open('adb labels/{}_labels.pkl'.format(i), 'wb') as f:
        pickle.dump(labels_symbol, f)
    with open('adb label locations/{}_label_locations.pkl'.format(i), 'wb') as f:
        pickle.dump(peaks, f)
"""
