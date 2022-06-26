import math
import os
import sys

from tqdm import tqdm

from definitions import DATA_DIR

_, dataset_name, n_splits_str = sys.argv
n_splits = int(n_splits_str)
data_dir = os.path.join(DATA_DIR, dataset_name, "fasta")

if not os.path.isdir(os.path.join(data_dir, "splits")):
    os.makedirs(os.path.join(data_dir, "splits"))
if os.path.isdir(os.path.join(data_dir, "splits", n_splits_str)):
    print(f"{n_splits_str} splits for dataset {dataset_name} already exist. Exiting...")
else:
    for subset in ["train", "test", "val"]:
        # prepare output directory and files
        os.makedirs(os.path.join(data_dir, "splits", n_splits_str, subset))
        out_files = [open(os.path.join(data_dir, "splits", n_splits_str, subset, f"{i}.csv"), "wt")
                     for i in range(n_splits)]

        with open(os.path.join(data_dir, f"{subset}.csv"), "rt") as in_file:
            # count lines, calculate split size and prepare files
            n_lines = sum(1 for _ in in_file)
            split_size = math.ceil(n_lines / n_splits)

            # read file from the beginning and split it into multiple smaller files
            in_file.seek(0)
            for i, line in tqdm(enumerate(in_file), desc=f"Splitting {subset}.csv", total=n_lines):
                out_files[i // split_size].write(line)

            # flush and close file writers
            for out_file in out_files:
                out_file.close()
