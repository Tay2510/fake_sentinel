import pandas as pd
from pathlib import Path

from fake_sentinel.data.utils.data_utils import clean_data
from fake_sentinel.data.paths import DFDC_DATAFRAME_FILE, DFDC_TRAIN_VIDEO_DIR, FACE_CROP_DIR


def load_dfdc_dataframe(metadata_file=DFDC_DATAFRAME_FILE, source_dir=DFDC_TRAIN_VIDEO_DIR):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: source_dir + x)

    return df


def load_crop_dataframe(metadata_file=DFDC_DATAFRAME_FILE, crop_dir=FACE_CROP_DIR):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: Path(crop_dir) / Path(x).stem)

    df = clean_data(df)

    return df
