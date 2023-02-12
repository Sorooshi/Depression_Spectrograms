import torch.nn as nn

from config.config import *


class DCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(1, 32, [7, 1]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([4, 3], stride=[1, 3]),

            nn.Conv2d(32, 64, [7, 1], stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1, 3]),
        )
        self.meta_features_ = nn.Sequential(
            nn.Linear(2, 16)
        )
        self.mixed_features_ = nn.Sequential(
            nn.Linear(33280 + 16, 256),
            nn.Linear(256, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, output_dim),
            # nn.Sigmoid() # for BCELoss
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, images, meta):
        images = self.image_features_(images)
        images = torch.flatten(images, 1) # flatten all dimensions except batch
        meta = self.meta_features_(meta)
        mixed = torch.cat((images, meta), 1)
        mixed = self.mixed_features_(mixed)
        return mixed
