import pickle
import numpy as np
from multiprocessing import Pool
from pathlib import Path

from fake_sentinel.data.split import NO_FACE_CROPS

CROP_DIR = '/home/jeremy/data/kaggle/dfdc_face_crops'


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


def save_pickle(instance, file_path):
    with Path(file_path).open('wb') as f:
        pickle.dump(instance, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with Path(file_path).open('rb') as f:
        instance = pickle.load(f)

    return instance


def chunks(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


def parallelize_dataframe(df, func, n_cores=4):
    if len(df) == 0:
        return
    else:
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        pool.map(func, df_split)
        pool.close()
        pool.join()
