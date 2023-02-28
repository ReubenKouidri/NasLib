import torch.nn as nn


class DenseBlock(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: bool | None = True
    ) -> None:
        super(DenseBlock, self).__init__(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=0.5) if dropout else None
        )

