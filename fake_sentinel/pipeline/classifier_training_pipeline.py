import sys
import argparse
import json
import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader

from fake_sentinel.data.query import load_crop_dataframe, split_train_val
from fake_sentinel.data.loading.dataset import FaceCropDataset
from fake_sentinel.data.loading.sampler import BatchSampler
from fake_sentinel.model.classifier import create_classifier
from fake_sentinel.train.trainer import train_model
from fake_sentinel.pipeline.configs import CONFIGS
from fake_sentinel.evaluation.evaulator import evaluate
from fake_sentinel.report.report_writer import write_notebook_report


def run_pipeline(test_mode=False, result_dir='result_dir', num_epochs=CONFIGS['EPOCHS'], eval_fraction=1.0):
    report_data = {}
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=False)
    model_path = result_dir / 'weights.pth'
    history_path = result_dir / 'history.json'
    report_path = result_dir / 'report.ipynb'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nUsing device:', device)

    # Data
    print('\nLoading Data...')
    df = load_crop_dataframe()

    train_df, val_df = split_train_val(df, mode=CONFIGS['VAL_MODE'], val_fraction=CONFIGS['VAL_FRACTION'], seed=CONFIGS['VAL_SEED'])

    train_dataset = FaceCropDataset(train_df, 'train', smoothing_epsilon=CONFIGS['SMOOTHING_EPSILON'])
    val_dataset = FaceCropDataset(val_df, 'val', smoothing_epsilon=CONFIGS['SMOOTHING_EPSILON'])

    if test_mode:
        train_dataset.real_indices = train_dataset.real_indices[:1000]
        val_dataset.real_indices = val_dataset.real_indices[:100]
        num_epochs = 5
        eval_fraction = 0.05

    train_sampler = BatchSampler(train_dataset, CONFIGS['BACKWARD_BATCH_SIZE'], shuffle=True)
    val_sampler = BatchSampler(val_dataset, CONFIGS['FORWARD_BATCH_SIZE'], shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    print('Originals: Train = {:,} | Val = {:,}'.format(len(train_dataset.real_indices),
                                                        len(val_dataset.real_indices)))

    # Model
    print('\nCreating Model...')
    model = create_classifier(model_name=CONFIGS['MODEL_NAME'],
                              pretrained=CONFIGS['PRETRAINED'],
                              freeze_features=CONFIGS['FREEZE_FEATURES'])
    model.to(device)

    params_to_update = model.parameters()
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.SGD(params_to_update, lr=CONFIGS['INITIAL_LR'],
                                momentum=CONFIGS['MOMENTUM'], weight_decay=CONFIGS['L2_REGULARIZATION'])

    # Training
    print('\nTraining...')
    model, history = train_model(model=model, dataloaders={'train': train_loader, 'val': val_loader}, device=device,
                                 criterion=criterion, optimizer=optimizer, save_path=model_path, num_epochs=num_epochs)

    with open(str(history_path), 'w') as f:
        json.dump(history, f)

    # Evaluation
    print('\nEvaluating...')
    since = time.time()
    log_loss = evaluate(str(model_path), eval_fraction=eval_fraction)
    avg_time = (time.time() - since) / (400 * eval_fraction)
    report_data['eval_loss'] = log_loss
    report_data['eval_time'] = avg_time
    print('Log loss:', log_loss)

    # Report
    print('\nGenerating Report...')
    write_notebook_report(result_dir, report_data, report_path)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--test_mode', action='store_true', help='Run in test mode (use less data)')
    parser.add_argument('-d', '--directory', default='result_dir', required=False, help='Directory to save training results')

    args = parser.parse_args(argv)

    run_pipeline(test_mode=args.test_mode, result_dir=args.directory)


if __name__ == '__main__':
    main(sys.argv[1:])
