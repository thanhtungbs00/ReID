import pytorch_lightning as pl

import src


class DogCatClassifier(pl.LightningModule):

    # Data processing
    def prepare_data(self):
        return src.data.prepare_data(self)

    def train_dataloader(self):
        return NotImplementedError

    def val_dataloader(self):
        return NotImplementedError

    def test_dataloader(self):
        return NotImplementedError

    # Layers specification and feeding process
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        src.models.model.define_layers(self)

    def forward(self, x):
        return src.models.model.forward(self, x)

    # Optimizer
    def configure_optimizers(self):
        return NotImplementedError

    # Training and validation specification
    def training_step(self, *args, **kwargs):
        return NotImplementedError

    def training_step_end(self, *args, **kwargs):
        return NotImplementedError

    def validation_step(self, *args, **kwargs):
        return NotImplementedError

    def validation_epoch_end(self, outputs):
        return NotImplementedError


def main():

    # Step 1: Prepare data
    download()
    return 0


if __name__ == "__main__":
    main()
