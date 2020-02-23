import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from fake_sentinel.data.query import load_crop_dataframe
from fake_sentinel.data.loading.dataset import FaceCropDataset
from fake_sentinel.model.classifier import create_classifier
from fake_sentinel.train.helpers import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data
df = load_crop_dataframe()

train_df = df[:1000]
val_df = df[1000:1100]

train_dataset = FaceCropDataset(train_df, 'train')
val_dataset = FaceCropDataset(val_df, 'val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model
model = create_classifier(pretrained=False)
model.to(device)

summary(model, (3, 299, 299))

criterion = torch.nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Training
train_model(model=model,
            dataloaders={'train': train_loader, 'val': val_loader},
            criterion=criterion,
            optimizer=optimizer,
            device=device)
