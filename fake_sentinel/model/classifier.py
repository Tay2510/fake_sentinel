import torch.nn as nn
from fake_sentinel.model.cnn.xception import xception


def create_classifier(pretrained=True, freeze_features=False):
    if pretrained:
        pretrained_data = 'imagenet'
    else:
        pretrained_data = None

    model = xception(num_classes=1000, pretrained=pretrained_data)

    if freeze_features:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.conv3.parameters():
            parameter.requires_grad = True

        for parameter in model.conv4.parameters():
            parameter.requires_grad = True

    # Handle the primary net
    num_features = model.last_linear.in_features
    model.last_linear = nn.Linear(num_features, 1)

    return model
