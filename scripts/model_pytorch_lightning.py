# %%
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pytorch_lightning as pl
import random
from pathlib import Path
from PIL import Image

# os.chdir('DeepLearning/YandexGPU/OCR Keras Jetson')

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


# %%
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


model = ConvNet()

trainer = pl.Trainer(gpus=0, max_epochs=20, progress_bar_refresh_rate=20)
trainer.fit(model, dataloaders['train'], dataloaders['val'])

# %%
start_time = time.time()
train_dir = 'data/val'
img_height = 32
img_width = 32
for i in range(33):
    folder = random.choice(os.listdir(train_dir))
    file = random.choice([f for f in os.listdir(train_dir + '/' + folder)])

    path_to_file = Path().joinpath(train_dir, folder, file)
    img = Image.open(path_to_file)
    img_array = data_transforms['val'](img)
    img_array = img_array.view(1, 3, 64, 64)

    outputs = model(img_array)
    _, preds = torch.max(outputs, 1)

    print("This image belongs to {}. True value {}".format(class_names[preds], folder))
print(f'Process time {round(time.time() - start_time, 3)}, sec.')
# %% Save model
PATH = "model/pytorch_model/pytorch_model.pt"
torch.save(model.state_dict(), PATH)