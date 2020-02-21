import pandas as pd
from pathlib import Path

from fake_sentinel.data.data_utils import clean_data

SOURCE_DIR = '/home/jeremy/data/kaggle/dfdc/train/'
CROP_DIR = '/home/jeremy/data/kaggle/dfdc_face_crops'

METADATA_FILE = Path(__file__).parent / 'dfdc.csv'


def load_dfdc_dataframe(metadata_file=METADATA_FILE, source_dir=SOURCE_DIR):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: source_dir + x)

    return df


def load_crop_dataframe(metadata_file=METADATA_FILE, crop_dir=CROP_DIR):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: Path(crop_dir) / Path(x).stem)

    df = clean_data(df)

    return df
