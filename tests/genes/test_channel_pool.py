import unittest
import torch

from dnasty.components import ChannelPool


class TestChannelPool(unittest.TestCase):
    def test_channel_pooling(self):
        channel_pool = ChannelPool()

        # Create a test input tensor of shape (N, C, H, W)
        N, C, H, W = 16, 64, 32, 32
        x = torch.randn(N, C, H, W)

        y = channel_pool.forward(x)
        self.assertEqual(y.shape, (N, 2, H, W))

        # Verify that the first channel is the max and the second is the mean
        x_max = torch.max(x, dim=1)[0]
        x_mean = torch.mean(x, dim=1)
        self.assertTrue(torch.allclose(y[:, 0, :, :], x_max))
        self.assertTrue(torch.allclose(y[:, 1, :, :], x_mean))


if __name__ == '__main__':
    unittest.main()
