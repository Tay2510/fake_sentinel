import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from fake_sentinel.data.query import load_crop_dataframe
from fake_sentinel.data.loading.dataset import FaceCropDataset
from fake_sentinel.model.classifier import create_classifier
from fake_sentinel.train.helpers import train_model
from fake_sentinel.pipeline.configs import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Data
print('Loading Data...')
df = load_crop_dataframe()

train_df = df[:3000]
val_df = df[3000:3500]

train_dataset = FaceCropDataset(train_df, 'train')
val_dataset = FaceCropDataset(val_df, 'val')

train_loader = DataLoader(train_dataset, batch_size=BACKWARD_BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=FORWARD_BATCH_SIZE, shuffle=False, num_workers=4)

# Model
print('Creating Model...')
model = create_classifier(pretrained=False)
model.to(device)

summary(model, (3, 299, 299))

criterion = torch.nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = torch.optim.SGD(params_to_update, lr=INITIAL_LR, momentum=MOMENTUM)

# Training
print('Training...')
model, history = train_model(model=model, dataloaders={'train': train_loader, 'val': val_loader},
                             criterion=criterion, optimizer=optimizer, device=device)