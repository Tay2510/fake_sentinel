import torch.nn as nn
from fake_sentinel.model.cnn.xception import xception


def create_classifier(pretrained=True):
    if pretrained:
        pretrained_data = 'imagenet'
    else:
        pretrained_data = None

    model = xception(num_classes=1000, pretrained=pretrained_data)

    # Handle the primary net
    num_features = model.last_linear.in_features
    model.last_linear = nn.Linear(num_features, 1)

    return model
