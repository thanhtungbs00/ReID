import torch
import pytorch_lightning as pl
import src


class Classifier(pl.LightningModule):

    # Layers specification and feeding process
    def __init__(self):
        super(Classifier, self).__init__()
        src.models.define_layers(self)

    def forward(self, x):
        return src.models.model.forward(self, x)

    # Optimizer
    def configure_optimizers(self):
        return src.optimizers.configure_optimizers(self)

    # Data processing
    def prepare_data(self):
        return src.data.prepare_data(self)

    def train_dataloader(self):
        return src.data.train_loader(self)

    def val_dataloader(self):
        return src.data.val_loader(self)

    def test_dataloader(self):
        return src.data.test_loader(self)

    # Training and validation specification

    def training_step(self, *args, **kwargs):
        return NotImplemented

    def training_step_end(self, *args, **kwargs):
        return NotImplemented

    def validation_step(self, *args, **kwargs):
        return NotImplemented

    def validation_epoch_end(self, outputs):
        return NotImplemented


def main():

    # Step 1: Prepare data
    classifier = Classifier()
    classifier.prepare_data()
    dataloader = torch.utils.data.DataLoader(
        classifier.training_set, batch_size=2)
    for i in dataloader:
        print(i)
        break
    return 0


if __name__ == "__main__":
    main()
