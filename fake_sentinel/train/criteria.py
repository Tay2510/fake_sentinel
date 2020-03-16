import torch

from fake_sentinel.pipeline.configs import *


def get_criteria(model):
    params_to_update = model.parameters()
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.SGD(params_to_update, lr=INITIAL_LR, momentum=MOMENTUM)

    return criterion, optimizer
