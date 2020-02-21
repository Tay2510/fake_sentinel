from pathlib import Path
from fake_sentinel.data.split import NO_FACE_CROPS


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
