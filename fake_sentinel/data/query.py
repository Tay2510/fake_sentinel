import pandas as pd
from pathlib import Path

from fake_sentinel.data.split import NO_FACE_CROPS

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


def clean_data(dataframe):
    return dataframe.loc[dataframe.index.difference(NO_FACE_CROPS)]


def count_faces(dir_name):
    return len([str(f.name) for f in Path(dir_name).iterdir() if f.is_dir()])


def build_crop_table(dir_name):
    face_ids = [str(f.name) for f in Path(dir_name).iterdir() if f.is_dir()]
    table = {}
    for i in face_ids:
        crop_list = list((Path(dir_name) / i).glob('**/*.png'))
        table[int(i)] = [str(p.name) for p in crop_list]

    return table
