import pytorch_lightning as pl
import torch.nn as nn
import torch


def define_layers(self):
    # input_shape: 256x256x3
    self.layer1 = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 128x128x8
    self.layer2 = nn.Sequential(
        nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 64x64x16
    self.layer3 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 32x32x32
    self.layer4 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 16x16x64
    self.layer5 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 8x8x128
    self.layer6 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 4x4x128
    self.layer7 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)),
        nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 2x2x256
    self.layer8 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(2, 2),
            stride=(1, 1)),
        nn.ReLU())
    # 1x1x256
    # view: 256
    self.fc1 = nn.Sequential(
        nn.Linear(
            in_features=256,
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
            out_features=2
        ),
        nn.Sigmoid())


def forward(self, x):
    bsize, w, h, c = x.size()
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = x.view(bsize, 256)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

# class DogCatClassifier(pl.LightningModule):

#     def __init__(self, dataset, dataloader, test_set=None):
#         super(DogCatClassifier, self).__init__()
#         self.dataset = dataset
#         self.dataloader = dataloader
#         self.test_set = test_set

#         # Layers specification
#         # input_shape: 256x256x3
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=8,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 128x128x8
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=8,
#                 out_channels=16,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 64x64x16
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 32x32x32
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=64,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 16x16x64
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 8x8x128
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=128,
#                 out_channels=128,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 4x4x128
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=128,
#                 out_channels=128,
#                 kernel_size=(2, 2),
#                 stride=(1, 1),
#                 padding=(1, 1)),
#             nn.MaxPool2d(
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 2x2x256
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=128,
#                 out_channels=256,
#                 kernel_size=(2, 2),
#                 stride=(1, 1)),
#             nn.ReLU())
#         # 1x1x256
#         # view: 256
#         self.fc1 = nn.Sequential(
#             nn.Linear(
#                 in_features=256,
#                 out_features=64
#             ),
#             nn.ReLU())
#         # 64
#         self.fc2 = nn.Sequential(
#             nn.Linear(
#                 in_features=64,
#                 out_features=16
#             ),
#             nn.ReLU())
#         # 16
#         self.fc3 = nn.Sequential(
#             nn.Linear(
#                 in_features=16,
#                 out_features=2
#             ),
#             nn.Sigmoid())

#     def forward(self, x):
#

#     def prepare_data(self):
#         print("Preparing data")
#         validation_rate = 0.1

#         t = int(len(dataset) * validation_rate)
#         self.training_set, self.validation_set = torch.utils.data.random_split(
#             len(dataset)-t, t)

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.training_set, batch_size=10)

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.validation_set, batch_size=10)

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.test_set, batch_size=10)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         # we already defined forward and loss in the lightning module. We'll show the full code next
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)

#         logs = {'train_loss': loss}
#         return {'loss': loss, 'log': logs}

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         return {'val_loss': loss}

#     def validation_end(self, outputs):
#         # outputs is an array with what you returned in validation_step for each batch
#         # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'val_loss': avg_loss}
#         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
