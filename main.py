import torch
import pytorch_lightning as pl
import src


class Classifier(pl.LightningModule):

    # Layers specification and feeding process
    def __init__(self):
        super(Classifier, self).__init__()
        src.models.define_layers(self)
        self.configure_loss()

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

    # Loss function
    def configure_loss(self):
        return src.losses.configure_loss(self)

    # Training and validation progresses  specification

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_predict = self.forward(x)
        loss = self.cross_entropy(y_predict, y)
        logs = {
            "train_loss": loss
        }
        return {
            "loss": loss,
            "log": logs
        }

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_predict = self.forward(x)
        loss = self.cross_entropy(y_predict, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


def main():

    # Step 1: Prepare data
    classifier = Classifier()

    # classifier.prepare_data()
    # dataloader = classifier.train_dataloader()
    # for i in dataloader:
    #     x, y = i
    #     x = x.float
    #     print(x)
    #     yp = classifier.forward(x)
    #     print(yp.size())

    trainer = pl.Trainer(check_val_every_n_epoch=100)
    trainer.fit(classifier)
    return 0


if __name__ == "__main__":
    main()
