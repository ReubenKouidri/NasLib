from scipy.io import loadmat
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pywt
import numpy as np
from tsmoothie import ConvolutionSmoother
from typing import Optional, Tuple, Iterable, Union
import matplotlib.pyplot as plt
from wavelets import cmor, mexh


class CPSCDataset(Dataset):
    AR_classes = {"SR": 1, "AF": 2, "I-AVB": 3, "LBBB": 4, "RBBB": 5, "PAC": 6, "PVC": 7, "STD": 8, "STE": 9}
    LENGTH = 4 * 500  # 4s at sampling frequency of 500Hz

    def __init__(
            self,
            data_path: str,
            reference_path: str,
            normalize: Optional[bool] = True,
            smoothen: Optional[bool] = True,
            trim: Optional[bool] = True,
            lead: Optional[Union[int, Iterable]] = 0,
            test: Optional[bool] = False
    ):
        super(CPSCDataset, self).__init__()
        self.test = test
        self.data_path = data_path
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.lead = torch.tensor(lead)
        self.references = pd.read_csv(reference_path)
        self.names = self.references['Recording']
        self.targets = torch.as_tensor(self.references['First_label'], dtype=torch.int8)  # [1-9], int8 as only 1-9

    @staticmethod
    def _normalize(data):
        return (data - min(data)) / (max(data) - min(data))

    @staticmethod
    def _trim_data(data, size, step=1):
        return data[:size:step]

    @staticmethod
    def _smoothen(data):
        smoother = ConvolutionSmoother(window_len=8, window_type='ones')
        smoother.smooth(data)
        return smoother.smooth_data[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
            - ECG is cut to 2000 data points (4 seconds) (if True)
            - ECG is smoothed (if True)
            - returns a tuple: (img, target)
        """
        file_path = os.path.join(self.data_path, self.references.iloc[item, 0])
        data = loadmat(f'{file_path}.mat')
        ecg_data = data['ECG']['data'][0][0][self.lead - 1]  # leads in [1,12] hence -1 indexes correctly

        step = 2
        base = self._trim_data(ecg_data, self.LENGTH, step) if self.trim else ecg_data
        base = self._normalize(base) if self.normalize else base
        base = self._smoothen(base) if self.smoothen else base

        ecg_data = torch.as_tensor(base, dtype=torch.float32)

        if self.test:
            return ecg_data, self.names[item]
        return ecg_data, self.targets[item]

    def __len__(self):
        """
            Length of the dataset
        """
        return len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])


class CPSCDataset2D(CPSCDataset):
    wavelets = {"mexh": 64, "cmor": 64}

    def __init__(
        self,
        data_path: str,
        reference_path: str,
        wavelet: Optional[str] = "mexh",
    ):
        super(CPSCDataset2D, self).__init__(data_path, reference_path)
        self.wavelet = wavelet if self.wavelets.__contains__(wavelet) else "mexh"

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ecg, tgt = CPSCDataset.__getitem__(self, item)
        ecg = np.array(ecg)
        ecg_img = eval(self.wavelet)(ecg, self.wavelet, self.wavelets[self.wavelet])
        ecg_img = torch.as_tensor(ecg_img)

        if self.test:
            return ecg_img, self.names[item]
        return ecg_img, self.targets[item]
