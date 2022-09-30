import os
import random
import sys

from tqdm import tqdm

from definitions import DATA_DIR
from src.samplers.sampler_enum import SamplerEnum

_, autoencoder_name = sys.argv
embeddings_dir = os.path.join(DATA_DIR, "deepmicrobes_mag_reads", "embeddings", autoencoder_name,
                              SamplerEnum.NO_SAMPLER.value)

# read embeddings
embeddings = []
for filename in tqdm(os.listdir(embeddings_dir), desc="Reading embeddings"):
    if filename not in ["train.csv", "test.csv", "val.csv"]:
        with open(os.path.join(embeddings_dir, filename), "rt") as file:
            for line in file:
                embeddings.append(line.strip())

# shuffle embeddings and write into subset files
random.shuffle(embeddings)
train_file = open(os.path.join(embeddings_dir, "train.csv"), "wt")
test_file = open(os.path.join(embeddings_dir, "test.csv"), "wt")
val_file = open(os.path.join(embeddings_dir, "val.csv"), "wt")
for i, embedding in tqdm(enumerate(embeddings), total=len(embeddings), desc="Creating subsets"):
    if i < 0.8 * len(embeddings):
        train_file.write(embedding + "\n")
    elif 0.8 * len(embeddings) < i < 0.9 * len(embeddings):
        test_file.write(embedding + "\n")
    else:
        val_file.write(embedding + "\n")

train_file.close()
test_file.close()
val_file.close()
