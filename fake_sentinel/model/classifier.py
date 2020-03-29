import torch.nn as nn
from torchvision.models import resnext50_32x4d as resnext50
from torchvision.models import resnext101_32x8d as resnext101

from fake_sentinel.model.cnn.xception import xception
from fake_sentinel.model.cnn.hrnet_cls import hrnet_classifier


def create_classifier(model_name='xception', pretrained=True, freeze_features=False):
    model = get_basic_model(model_name, pretrained=pretrained, freeze_features=freeze_features)

    return model


def get_basic_model(model_name, pretrained=True, freeze_features=False):
    if model_name == 'xception':
        model = xception(pretrained=pretrained)

        if freeze_features:
            for parameter in model.parameters():
                parameter.requires_grad = False

            for parameter in model.conv3.parameters():
                parameter.requires_grad = True

            for parameter in model.conv4.parameters():
                parameter.requires_grad = True

    elif model_name == 'resnext50':
        model = resnext50(pretrained=pretrained)

    elif model_name == 'resnext101':
        model = resnext101(pretrained=pretrained)

    elif model_name == 'hrnet':
        model = hrnet_classifier(pretrained=pretrained, freeze_features=freeze_features)

    else:
        raise NotImplementedError

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
