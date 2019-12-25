import pandas as pd
from pathlib import Path

METADATA_FILE = Path(__file__).parent / 'dfdc.csv'


def load_dfdc_dataframe(metadata_file=METADATA_FILE):
    df = pd.read_csv(metadata_file, index_col='index')
    df = clean_data(df)

    return df


def clean_data(df):
    sample_without_video_files = [
        'wipjitfmta',
        'wpuxmawbkj',
        'pvohowzowy',
        'innmztffzd',
        'cfxiikrhep',
        'dzjjtfwiqc',
        'zzfhqvpsyp',
        'glleqxulnn'
    ]
    return df.drop(sample_without_video_files)
