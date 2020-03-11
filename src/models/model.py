import pytorch_lightning as pl
import torch.nn as nn
import torch


def define_layers(self):
    # input_shape: 3x32x32
    self.layer1 = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2)),
        nn.ReLU())
    # 8x16x16
    self.layer2 = nn.Sequential(
        nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2)),
        nn.ReLU())
    # 16x8x8
    self.layer3 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2)),
        nn.ReLU())
    # 32x4x4
    self.layer4 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2)),
        nn.ReLU())
    # 64x2x2
    self.layer5 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2)),
        nn.ReLU())
    # 128x1x1
    # view: 128
    self.fc1 = nn.Sequential(
        nn.Linear(
            in_features=128,
            out_features=64
        ),
        nn.ReLU())
    # 64
    self.fc2 = nn.Sequential(
        nn.Linear(
            in_features=64,
            out_features=16
        ),
        nn.ReLU())
    # 16
    self.fc3 = nn.Sequential(
        nn.Linear(
            in_features=16,
            out_features=10
        ),
        nn.Sigmoid())


def forward(self, x):
    x = x.float()
    bsize, w, h, c = x.size()
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(bsize, 128)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
