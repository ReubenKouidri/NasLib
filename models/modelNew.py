import dnasty.components as components
import torch.nn as nn


class Model(nn.Sequential):
    def __init__(self):
        super(Model, self).__init__()

    def build(self) -> None: ...
