import sys
import argparse
import pandas as pd
from pathlib import Path

LABEL_NAME = 'metadata.json'


def create_dfdc_dataframe(sub_dir_list):
    df = None

    for d in sub_dir_list:
        file = d / LABEL_NAME

        if df is None:
            df = load_partial_dataframe(file)
        else:
            df = pd.concat([df, load_partial_dataframe(file)], ignore_index=True)

    df['filename'] = df['index'].apply(lambda x: x + '.mp4')

    df = df.set_index('index')

    df = clean_data(df)

    return df


def load_partial_dataframe(label_file):
    df = pd.read_json(label_file)
    df = df.transpose()
    df = df.reset_index()
    df['index'] = df['index'].apply(lambda x: x.replace('.mp4', ''))

    def simplify(x):
        if isinstance(x, str):
            return x.replace('.mp4', '')
        else:
            return x

    df['original'] = df['original'].apply(lambda x: simplify(x))

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


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', help='Root path to 50 unzip train data folders.', required=True)
    parser.add_argument('-f', '--save_file', help='Filename for the final csv file.', default='dfdc.csv', required=False)

    args = parser.parse_args(argv)

    data_subdirs = [x for x in Path(args.data_dir).iterdir()]

    data = create_dfdc_dataframe(data_subdirs)

    data.to_csv(args.save_file)


if __name__ == '__main__':
    main(sys.argv[1:])
