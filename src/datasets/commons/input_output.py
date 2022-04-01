import gzip
import pickle


def read_from_disk(data_dir, filename):
    with gzip.open(f"{data_dir}/{filename}", "rb") as file:
        record = pickle.load(file)
    return record


def save_to_disk(data_dir, filename, object):
    with gzip.open(f"{data_dir}/{filename}", "wb") as file:
        pickle.dump(object, file)
