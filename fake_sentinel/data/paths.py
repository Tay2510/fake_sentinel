from pathlib import Path

DATASET_DIR = Path(__file__).parents[2] / 'datasets'

PREPROCESSED_FILE_DIR = Path(__file__).parent / 'resources'


DFDC_DATAFRAME_FILE = PREPROCESSED_FILE_DIR / 'dfdc.csv'

FACE_CROP_TABLE = PREPROCESSED_FILE_DIR / 'crop_table.pkl'

VAL_SPLIT_LIST = PREPROCESSED_FILE_DIR / 'val_splits.pkl'


# symbolic links to the datasets
DFDC_TRAIN_VIDEO_DIR = DATASET_DIR / 'videos'

FACENET_RESULT_DIR = DATASET_DIR / 'facenet_results'

FACE_CROP_DIR = DATASET_DIR / 'face_crops'
