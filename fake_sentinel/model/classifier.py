import torch.nn as nn
from fake_sentinel.model.cnn.inception import inception_v3


def create_classifier(num_classes=2, pretrained=True):
    model = inception_v3(pretrained=pretrained)

    # Handle the auxilary net
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_features, num_classes)

    # Handle the primary net
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
