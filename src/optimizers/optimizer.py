import torch.optim as optim

lr = 0.01


def configure_optimizers(self):
    return optim.Adam(self.parameters())
