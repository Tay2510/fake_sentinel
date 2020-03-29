from torchvision import transforms


CONFIGS = {
    # Data
    'VAL_MODE': 'random',  # 'random' / 'chunk'
    'VAL_FRACTION': 0.1,  # P.S. behavior differently when using 'chunk' mode
    'VAL_SEED': 1337,

    # Batching
    'BACKWARD_BATCH_SIZE': 32,
    'FORWARD_BATCH_SIZE': 128,

    # Model
    'MODEL_NAME': 'resnext50',
    'PRETRAINED': True,
    'TRAIN_LOSS': 'BCE',  # 'BCE' / 'Focal'

    # Generalization
    'FREEZE_FEATURES': False,
    'L2_REGULARIZATION': 0,
    'SMOOTHING_EPSILON': 0,
    'MIXED_UP': -1,  # use float number within (0, inf) to activate mixed-up
    'AUGMENTATION': [transforms.RandomHorizontalFlip()],

    # Training
    'OPTIMIZER': 'SGD',  # 'SGD' / 'Adabound'
    'EPOCHS': 12,
    'INITIAL_LR': 0.004,
    'MOMENTUM': 0.85
}


INPUT_SHAPES = {
    'xception': (299, 299),
    'hrnet': (256, 256),
    'resnext50': (224, 224),
    'resnext101': (224, 224)
}
