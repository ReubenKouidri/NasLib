import torch.nn as nn


def get_activation(activation) -> nn.Module:
    if activation is None:
        return nn.ReLU()
    elif isinstance(activation, str):
        activation_module = getattr(nn, activation)()
    elif isinstance(activation, nn.Module):
        activation_module = activation
    else:
        raise TypeError("Activation must be a string, nn.Module, or None.")
    return activation_module
