import pickle
from pathlib import Path


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
