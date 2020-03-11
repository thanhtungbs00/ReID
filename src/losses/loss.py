from torch.nn import functional as F


def configure_loss(self):
    self.cross_entropy = F.nll_loss
