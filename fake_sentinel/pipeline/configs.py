CONFIGS = {
    # Data
    'VAL_MODE': 'random',  # 'random' / 'chunk'
    'VAL_FRACTION': 1.0,
    'VAL_SEED': 1337,

    # Batching
    'BACKWARD_BATCH_SIZE': 32,
    'FORWARD_BATCH_SIZE': 128,

    # Model
    'MODEL_NAME': 'xception',
    'PRETRAINED': True,

    # Generalization
    'FREEZE_FEATURES': False,
    'L2_REGULARIZATION': 0,
    'SMOOTHING_EPSILON': 0,

    # Training
    'EPOCHS': 12,
    'INITIAL_LR': 0.004,
    'MOMENTUM': 0.85
}
