import json
import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, MODELS_DIR
from src.datasets.commons import preprocessing
from src.samplers.hypercube_fingerprint_sampler import HypercubeFingerprintSampler


def make_sampled_embeddings(sequence, encoder, sampler):
    int_encoded = tf.constant(preprocessing.integer_encode(sequence), dtype=tf.uint8)
    one_hot_encoded = preprocessing.one_hot_encode_sequences(int_encoded, dtype=tf.uint8)
    windows = preprocessing.split_into_windows(one_hot_encoded, 128, 3)
    embeddings = tf.constant(encoder.predict(windows, batch_size=32))
    return sampler.sample_tf(embeddings)


# read arguments
_, dataset_dir, autoencoder_name, sampler_name, subset, n_processes_str, split_index_str = sys.argv
n_processes = int(n_processes_str)
split_index = int(split_index_str)

# prepare encoder and sampler
encoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoders", autoencoder_name)).encoder
# todo: calculate split values on fraction of dataset, like in make_embeddings.py, but move the code to a new script
# split_values = np.array([[2.33112024], [3.2449895], [0.97849213], [2.04137627], [1.55365889],
#                          [1.15269995], [1.27782217], [1.47252198], [1.28234643], [3.08180996]])
# todo: generalize to multiple split values per axis - see line above
split_values = np.array([2.33112024, 3.2449895, 0.97849213, 2.04137627, 1.55365889,
                         1.15269995, 1.27782217, 1.47252198, 1.28234643, 3.08180996])
# todo: parametrize sampler based on "sampler_name" input argument
sampler = HypercubeFingerprintSampler(encoder.output.shape[-1], split_values,
                                      name="hypercube-fingerprint-median-sampler")

subset_dir = os.path.join(DATA_DIR, dataset_dir, "sampled-embeddings", autoencoder_name, sampler_name, "splits",
                          n_processes_str, subset)
try:
    os.makedirs(subset_dir)
except FileExistsError:
    # another process already created the output directory
    pass
in_file_path = os.path.join(DATA_DIR, dataset_dir, "fasta", "splits", n_processes_str, subset, f"{split_index}.csv")
out_file_path = os.path.join(DATA_DIR, dataset_dir, "sampled-embeddings", autoencoder_name, sampler_name,
                             "splits", n_processes_str, subset, f"{split_index}.csv")

with open(in_file_path, "rt") as in_file, open(out_file_path, "wt") as out_file:
    for i, line in enumerate(in_file):
        organism_id, sequence = line.strip().split(",")
        try:
            sampled_embeddings = make_sampled_embeddings(sequence, encoder, sampler)
            out_file.write(f"{organism_id};{json.dumps(sampled_embeddings.numpy().tolist())}\n")
        except tf.errors.ResourceExhaustedError as e:
            print("Sequence was too long to fit in memory. Continuing with next one...")

        if i % 100 == 0:
            out_file.flush()
