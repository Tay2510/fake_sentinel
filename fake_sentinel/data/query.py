import random
import pickle
import pandas as pd
from pathlib import Path

from fake_sentinel.data.tags import NO_FACE_CROPS
from fake_sentinel.data.paths import DFDC_DATAFRAME_FILE, DFDC_TRAIN_VIDEO_DIR, FACE_CROP_DIR, VAL_SPLIT_LIST


def load_dfdc_dataframe(metadata_file=DFDC_DATAFRAME_FILE, source_dir=DFDC_TRAIN_VIDEO_DIR, replace_nan=True):
    df = pd.read_csv(metadata_file)

    df['filename'] = df['filename'].apply(lambda x: Path(source_dir) / x)

    if replace_nan:
        replace_nan_with_id(df)

    return df


def load_crop_dataframe(metadata_file=DFDC_DATAFRAME_FILE, crop_dir=FACE_CROP_DIR, replace_nan=True):
    df = pd.read_csv(metadata_file)

    df['filename'] = df['filename'].apply(lambda x: Path(crop_dir) / Path(x).stem)

    df = clean_data(df)

    if replace_nan:
        replace_nan_with_id(df)

    return df


def split_train_val(df, mode='random', val_fraction=0.1, seed=1337):
    df_originals = get_originals(df)

    test_originals = list(get_originals(df[df.split == 'val'])['original'].unique())   # 200 REAL from public test
    train_originals = list(df_originals[~df_originals['original'].isin(test_originals)]['original'].unique())

    if mode == 'random':
        random.Random(seed).shuffle(train_originals)
        cut_off = int(val_fraction * len(train_originals))

        val_originals = train_originals[-cut_off:] + test_originals
        train_originals = train_originals[:-cut_off]

        df_train = df[df['original'].isin(train_originals)]
        df_val = df[df['original'].isin(val_originals)]

    elif mode == 'chunk':
        df_train = df[df['original'].isin(train_originals)]

        val_originals = get_val_originals()
        random.Random(seed).shuffle(val_originals)
        cut_off = int(val_fraction * len(val_originals))
        val_originals = val_originals[:cut_off]

        df_val = df_train[df_train['original'].isin(val_originals)]
        df_train = df_train[~df_train['original'].isin(val_originals)]

    else:
        raise NotImplementedError

    return df_train, df_val


def get_val_originals(val_splits_path=VAL_SPLIT_LIST):
    val_splits_list = pickle.load(Path(val_splits_path).open('rb'))
    return val_splits_list


def over_sampling_real_faces(df, factor=4):
    real = get_originals(df)
    return df.append([real] * factor, ignore_index=True)


def replace_nan_with_id(df):
    original_ids = df[df.original.isnull()]['index']
    df.loc[df['index'].isin(original_ids), 'original'] = original_ids


def get_originals(df):
    return df[df['label'] == 'REAL']


def clean_data(df):
    return df.loc[~df['index'].isin(NO_FACE_CROPS)]


def shuffle_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)
