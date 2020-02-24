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


def split_train_val(df, val_fraction=0.1):
    test_originals = get_originals(df[df.split == 'val']).index.to_list()   # 200 REAL from public test
    test_samples = df[df.original.isin(test_originals)].index.to_list()

    train_originals = get_originals(df.loc[df.index.difference(test_samples)]).index.to_list()

    cut_off = int(val_fraction * len(train_originals))

    val_originals = train_originals[-cut_off:] + test_originals
    train_originals = train_originals[:-cut_off]

    df_train = df[df.original.isin(train_originals)]
    df_val = df[df.original.isin(val_originals)]

    assert len(df_train) + len(df_val) == len(df)

    return df_train, df_val


def replace_nan_with_id(df):
    original_ids = df[df.original.isnull()].original.index
    df.loc[original_ids, 'original'] = original_ids


def get_originals(df):
    return df[df.index.to_series() == df.original]
