import sys
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from fake_sentinel.data.query import load_crop_dataframe, split_train_val
from fake_sentinel.data.loading.dataset import FaceCropDataset
from fake_sentinel.data.loading.sampler import BatchSampler
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

    train_df, val_df = split_train_val(df, val_fraction=VAL_FRACTION, seed=VAL_SEED)

    train_dataset = FaceCropDataset(train_df, 'train')
    val_dataset = FaceCropDataset(val_df, 'val')

    if test_mode:
        train_dataset.real_indices = train_dataset.real_indices[:1000]
        val_dataset.real_indices = val_dataset.real_indices[:100]
        num_epochs = 5

    train_sampler = BatchSampler(train_dataset, BACKWARD_BATCH_SIZE, shuffle=True)
    val_sampler = BatchSampler(val_dataset, FORWARD_BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    print('Originals: Train = {:,} | Val = {:,}'.format(len(train_dataset.real_indices),
                                                        len(val_dataset.real_indices)))

    # Model
    print('\nCreating Model...')
    model = create_classifier(pretrained=True)
    model.to(device)

    # Training
    print('\nTraining...')
    model, history = train_model(model=model, dataloaders={'train': train_loader, 'val': val_loader},
                                 device=device, result_dir=result_dir, num_epochs=num_epochs)

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
