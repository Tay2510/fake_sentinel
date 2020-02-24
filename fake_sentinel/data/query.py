import pandas as pd
from pathlib import Path

from fake_sentinel.data.utils.data_utils import clean_data
from fake_sentinel.data.paths import DFDC_DATAFRAME_FILE, DFDC_TRAIN_VIDEO_DIR, FACE_CROP_DIR


def load_dfdc_dataframe(metadata_file=DFDC_DATAFRAME_FILE, source_dir=DFDC_TRAIN_VIDEO_DIR, replace_nan=True):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: Path(source_dir) / x)

    if replace_nan:
        replace_nan_with_id(df)

    return df


def load_crop_dataframe(metadata_file=DFDC_DATAFRAME_FILE, crop_dir=FACE_CROP_DIR, replace_nan=True):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: Path(crop_dir) / Path(x).stem)

    df = clean_data(df)

    if replace_nan:
        replace_nan_with_id(df)

    return df


def replace_nan_with_id(df):
    original_ids = df[df.original.isnull()].original.index
    df.loc[original_ids, 'original'] = original_ids


def get_originals(df):
    return df[df.index.to_series() == df.original]
