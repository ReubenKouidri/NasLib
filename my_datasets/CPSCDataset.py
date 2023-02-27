from scipy.io import loadmat
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tsmoothie import ConvolutionSmoother
from typing import Iterable, Union, Any
import importlib


wavelets_module_name = "my_utils.wavelets"
wavelets_module = importlib.import_module(wavelets_module_name)


class CPSCDataset(Dataset):
    AR_classes = {"SR": 1, "AF": 2, "I-AVB": 3, "LBBB": 4, "RBBB": 5, "PAC": 6, "PVC": 7, "STD": 8, "STE": 9}
    LENGTH = 4 * 500  # 4s at sampling frequency of 500Hz
    base = ".."
    data_dir = "my_datasets/cpsc_data/test100"
    ref_dir = "my_datasets/cpsc_data/reference300.csv"

    def __init__(
            self,
            data_path: str | None = None,
            reference_path: str | None = None,
            normalize: bool | None = True,
            smoothen: bool | None = True,
            trim: bool | None = True,
            lead: Union[int, Iterable] | None = 3,
            test: bool | None = False
    ):
        super(CPSCDataset, self).__init__()
        self.test = test
        self.data_path = data_path if data_path is not None else os.path.join(self.base, self.data_dir)
        self.references = pd.read_csv(reference_path) if reference_path else pd.read_csv(os.path.join(self.base, self.ref_dir))
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.lead = torch.tensor(lead)
        self.names = self.references['Recording']
        self.targets = torch.as_tensor((self.references['First_label'] - 1)[:len(os.listdir(self.data_path))], dtype=torch.int64)  # [0-8]. !type long

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

    def __getitem__(self, item: int) -> tuple[Tensor, Any, Any] | tuple[Tensor, Any]:
        """
            - ECG is cut to 2000 data points (4 seconds) (if True)
            - ECG is smoothed (if True)
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
            return ecg_data, self.targets[item], self.names[item]
        return ecg_data, self.targets[item]

    def __len__(self):
        return len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])


class CPSCDataset2D(CPSCDataset):
    wavelets = {"mexh": 64, "cmor": 64}

    def __init__(
        self,
        data_path: str | None = None,
        reference_path: str | None = None,
        wavelet: str | None = "mexh",
        lead: int | None = 3
    ) -> None:
        super(CPSCDataset2D, self).__init__(data_path, reference_path, lead)
        self.wavelet = wavelet if self.wavelets.__contains__(wavelet) else "mexh"
        self.wavelet_fnc = getattr(wavelets_module, self.wavelet)

    def __getitem__(self, item: int) -> tuple[Any, tuple[Tensor, ...]]:
        out = super().__getitem__(item)
        ecg = np.array(out[0])
        ecg_img = self.wavelet_fnc(ecg, self.wavelets[self.wavelet])
        ecg_img = torch.as_tensor(ecg_img).unsqueeze(dim=0)
        return ecg_img, out[1:]
