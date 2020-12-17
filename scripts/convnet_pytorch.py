import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class ConvNet(pl.LightningModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.block_2 = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU(),
        )
        self.drop_out = nn.Dropout(0.2)
        self.output = nn.Linear(128, 36)

    def forward(self, x):
        out = self.block_1(x)
        out = self.drop_out(out)
        out = self.flatten(out)
        out = self.block_2(out)
        out = self.output(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        val_loss = criterion(y_hat, y)
        return val_loss
