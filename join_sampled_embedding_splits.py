import os
import sys

from tqdm import tqdm

from definitions import DATA_DIR

_, dataset_name, autoencoder_name, sampler_name, n_splits_str = sys.argv
n_splits = int(n_splits_str)
data_dir = os.path.join(DATA_DIR, dataset_name, "sampled-embeddings", autoencoder_name, sampler_name)

if not os.path.isdir(os.path.join(data_dir, "splits")):
    print(f"Sampled embeddings for given arguments do not exist. Exiting...")
else:
    for subset in ["train", "test", "val"]:
        # prepare directories and files
        in_dir_path = os.path.join(data_dir, "splits", n_splits_str, subset)
        out_file_path = os.path.join(data_dir, f"{subset}.csv")

        # join splits into single files
        with open(out_file_path, "wt") as out_file:
            for filename in tqdm(os.listdir(in_dir_path), desc=f"Creating {subset}.csv"):
                with open(os.path.join(in_dir_path, filename), "rt") as in_file:
                    for line in in_file:
                        out_file.write(line)
