import torch
from pathlib import Path

from .helpers import BlazeFace

DIRECTORY = Path(__file__).parent

WEIGHTS_PATH = DIRECTORY / 'weights.pth'
ANCHOR_PATH = DIRECTORY / 'anchors.npy'


def get_detector(wrights_path=WEIGHTS_PATH, anchor_path=ANCHOR_PATH, score_threshold=0.75, suppression_threshold=0.3):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Running on device: {device}'.format(device=device))

    net = BlazeFace().to(device)

    net.load_weights(wrights_path)
    net.load_anchors(anchor_path)

    # Optionally change the thresholds:
    net.min_score_thresh = score_threshold
    net.min_suppression_threshold = suppression_threshold

    return net
