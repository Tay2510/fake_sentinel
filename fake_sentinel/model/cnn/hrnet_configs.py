HRNET_CONFIGS = {
    'NUM_JOINTS': 19,

    'EXTRA': {
        'FINAL_CONV_KERNEL': 1,

        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': (4, 4),
            'NUM_CHANNELS': (18, 36),
            'FUSE_METHOD': 'SUM'
        },

        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': (4, 4, 4),
            'NUM_CHANNELS': (18, 36, 72),
            'FUSE_METHOD': 'SUM'
        },

        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': (4, 4, 4, 4),
            'NUM_CHANNELS': (18, 36, 72, 144),
            'FUSE_METHOD': 'SUM'
        },

    }
}