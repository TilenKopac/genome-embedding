import os
import random
import shutil

from tqdm import tqdm

data_dir = "../data/viruses/fasta"
subsets = ["train", "val", "test"]
split = [0.8, 0.1, 0.1]

for subset in subsets:
    os.makedirs(os.path.join(data_dir, subset))

filenames = [fname for fname in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, fname))]
random.shuffle(filenames)
n_records = len(filenames)

# train set
for filename in tqdm(filenames[:int(n_records * split[0])], desc="Creating train set"):
    shutil.move(os.path.join(data_dir, filename), os.path.join(data_dir, subsets[0], filename))

# validation set
for filename in tqdm(filenames[int(n_records * split[0]):int(n_records * sum(split[:2]))],
                     desc="Creating validation set"):
    shutil.move(os.path.join(data_dir, filename), os.path.join(data_dir, subsets[1], filename))

# test set
for filename in tqdm(filenames[int(n_records * sum(split[:2])):], desc="Creating test set"):
    shutil.move(os.path.join(data_dir, filename), os.path.join(data_dir, subsets[2], filename))
