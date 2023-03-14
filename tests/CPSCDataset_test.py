import unittest
import numpy as np
import torch
from my_utils.wavelets import mexh
from datasets.CPSCDataset import CPSCDataset, CPSCDataset2D

#TODO:
# add test cases for:
#  - _normalize(), _trim_data(), _smoothen()


class TestCPSCDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = CPSCDataset(data_path="../datasets/cpsc_data/test100",
                                   reference_path="../datasets/cpsc_data/reference300.csv",
                                   normalize=True,
                                   smoothen=True,
                                   trim=True,
                                   lead=3)

    def test_testing_mode(self):
        self.dataset.test = True
        x, y = self.dataset[0]
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertTrue(isinstance(y, torch.Tensor))
        self.assertEqual(len(y), 3)
        self.assertEqual(y[0].shape, torch.Size([]))
        self.assertEqual(y[1].shape, torch.Size([]))
        self.assertEqual(y[2].shape, torch.Size([]))

        self.dataset.test = False
        x, y = self.dataset[0]
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertTrue(isinstance(y, torch.Tensor))
        self.assertEqual(y.shape, torch.Size([]))

    def test_len(self):
        self.assertEqual(len(self.dataset), 100)

    def test_getitem(self):
        x, y = self.dataset[0]
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(x.shape, torch.Size([1000]))  # 1D Tensor of length 1000
        self.assertEqual(y.shape, torch.Size([]))  # 0-D Tensor


class TestCPSCDataset2D(unittest.TestCase):
    def setUp(self):
        self.dataset = CPSCDataset2D(data_path="../datasets/cpsc_data/test100",
                                     reference_path="../datasets/cpsc_data/reference300.csv",
                                     wavelet="mexh",
                                     lead=3)

    def test_testing_mode(self):
        self.dataset.test = True
        x0, y0 = self.dataset[0]
        self.assertTrue(isinstance(x0, torch.Tensor))
        self.assertEqual(x0.shape, torch.Size([1, 128, 128]))
        self.assertTrue(isinstance(y0, torch.Tensor))
        self.assertEqual(len(y0), 3)

        self.dataset.test = False
        x1, y1 = self.dataset[0]
        self.assertEqual(torch.all(x1), torch.all(x0))
        self.assertNotEqual(torch.all(y1), torch.all(y0))
        self.assertTrue(isinstance(x1, torch.Tensor))
        self.assertTrue(isinstance(y1, torch.Tensor))
        self.assertEqual(y1.shape, torch.Size([]))

    def test_getitem(self):
        x, y = self.dataset[0]
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(x.shape, torch.Size([1, 128, 128]))
        self.assertEqual(y.shape, torch.Size([]))  # 0-D Tensor

    def test_wavelet(self):
        data = np.random.rand(1000)
        img = mexh(data, 64)
        self.assertEqual(img.shape, (128, 128))
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertTrue(np.all(np.isfinite(img)))
