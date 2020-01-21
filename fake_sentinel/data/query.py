import pandas as pd
from pathlib import Path

SOURCE_DIR = '/home/jeremy/data/kaggle/dfdc/train/'

METADATA_FILE = Path(__file__).parent / 'dfdc.csv'


def load_dfdc_dataframe(metadata_file=METADATA_FILE, source_dir=SOURCE_DIR):
    df = pd.read_csv(metadata_file, index_col='index')

    df['filename'] = df['filename'].apply(lambda x: source_dir + x)

    return df
