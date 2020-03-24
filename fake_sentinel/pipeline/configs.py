CONFIGS = {
    # Data
    'VAL_MODE': 'random',  # 'random' / 'chunk'
    'VAL_FRACTION': 1.0,
    'VAL_SEED': 1337,

    # Batching
    'BACKWARD_BATCH_SIZE': 32,
    'FORWARD_BATCH_SIZE': 128,

    # Model
    'PRETRAINED': True,
    'FREEZE_FEATURES': False,

    # Training
    'EPOCHS': 12,
    'INITIAL_LR': 0.001,
    'MOMENTUM': 0.9
}
