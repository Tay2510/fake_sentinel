import sys
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from fake_sentinel.data.query import load_crop_dataframe, split_train_val, over_sampling_real_faces
from fake_sentinel.data.loading.dataset import FaceCropDataset
from fake_sentinel.model.classifier import create_classifier
from fake_sentinel.train.trainer import train_model
from fake_sentinel.pipeline.configs import *


def run_pipeline(test_mode=False, result_dir='result_dir', num_epochs=EPOCHS):
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nUsing device:', device)

    # Data
    print('\nLoading Data...')
    df = load_crop_dataframe()

    train_df, val_df = split_train_val(df, val_fraction=VAL_FRACTION)

    val_df = over_sampling_real_faces(val_df)

    if test_mode:
        train_df = train_df[:3000]
        val_df = val_df[:500]
        num_epochs = 5

    train_dataset = FaceCropDataset(train_df, 'train')
    val_dataset = FaceCropDataset(val_df, 'val')

    train_loader = DataLoader(train_dataset, batch_size=BACKWARD_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=FORWARD_BATCH_SIZE, shuffle=False, num_workers=4)

    print('Face Crops: Train = {:,} | Val = {:,}'.format(len(train_df), len(val_df)))

    # Model
    print('\nCreating Model...')
    model = create_classifier(pretrained=True)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = model.parameters()
    optimizer = torch.optim.SGD(params_to_update, lr=INITIAL_LR, momentum=MOMENTUM)

    # Training
    print('\nTraining...')
    model, history = train_model(model=model, dataloaders={'train': train_loader, 'val': val_loader},
                                 criterion=criterion, optimizer=optimizer, device=device,
                                 result_dir=result_dir, num_epochs=num_epochs)

    with open(str(result_dir / 'history.json'), 'w') as f:
        json.dump(history, f)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--test_mode', action='store_true', help='Run in test mode (use less data)')
    parser.add_argument('-d', '--directory', default='result_dir', required=False, help='Directory to save training results')

    args = parser.parse_args(argv)

    run_pipeline(test_mode=args.test_mode, result_dir=args.directory)


if __name__ == '__main__':
    main(sys.argv[1:])
