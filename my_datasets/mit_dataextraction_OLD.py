import wfdb
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os


"""
Start:
- get ECG signal
- get beat locations and annotations

Create labelled dataset:
- save each raw ECG record and its label info in a separate CSV file
- for each item:
    - get list of peaks
    - split up into 3-beat segments (between beats)
    - do not include first or last beats
    - 


"""

BASE_DIR = "/Users/juliettekouidri/Documents/Reuben/Projects/Python/NasLib/my_datasets/mit_data"
RECORDS_FN = "records.txt"
NAMES_PATH = os.path.join(BASE_DIR, RECORDS_FN)
FREQUENCY = 360


def get_names(records_path: str) -> list:
    with open(NAMES_PATH) as f:
        records = [record.strip() for record in f.readlines()]

    return records


def get_signal(signal_path: str):
    record = wfdb.rdrecord(signal_path, channels=[0])
    d_signal = record.adc()  # analogue to digital conversion
    d_signal = (d_signal - d_signal.min()) / (d_signal.max() - d_signal.min())  # normalise
    return d_signal


def extract_labels(filename):
    """
    - extract the labels and their locations (peaks) for each beat and
    - puts them into a lists
    - rdann reads in an annotation file and returns an Annotation object which has lists
    of beat locations and their associated labels
    """
    ann = wfdb.rdann(filename, 'atr', return_label_elements=['symbol'])
    # These two lines return the symbol and the locations
    label = ann.symbol
    loc = ann.sample
    return zip(label, loc)

names: list[int] = get_names(NAMES_PATH)

for i, name in enumerate(names):
    path_abs = os.path.join(BASE_DIR, str(name))
    signal = get_signal(path_abs)
    if i == 0 or i == 1:
        labels = tuple(extract_labels(path_abs))
        print(len(labels))

        """print(signal)
        x = np.linspace(0, FREQUENCY * 4, FREQUENCY * 4)
        signal = signal[: FREQUENCY * 4]
        plt.plot(x, signal)
        plt.show()"""
