import torch
from typing import Tuple


def gmp(x: torch.Tensor) -> torch.Tensor:
    """
    Applies global max pooling operation on a 4D tensor.

    Args:
        x (Tensor): Input tensor of shape (N, C, H, W).

    Returns:
        Tensor: Tensor of shape (N, C) representing the output of global max pooling operation applied
                on the input tensor x.
    """
    mp = torch.nn.AdaptiveMaxPool2d(output_size=1)(x)
    mp = mp.squeeze(dim=3).squeeze(dim=2)
    return mp


def create_random_tensor(shape: Tuple[int, int, int, int]) -> torch.Tensor:
    return torch.randn(shape)


# Create a random 4D tensor of shape (batch_size=1, channels=3, height=5, width=5)
x = torch.randn((1, 3, 5, 5))

# Apply global max pooling operation on the tensor x
output = gmp(x)
print(x)
print(output)
