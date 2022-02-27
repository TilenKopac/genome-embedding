import gzip
import pickle


def read_from_disk(data_dir, filename):
    with gzip.open(f"{data_dir}/{filename}", "rb") as file:
        record = pickle.load(file)
    return record
