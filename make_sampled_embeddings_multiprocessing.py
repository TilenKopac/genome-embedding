import json
import math
import multiprocessing
import os
import sys
from random import shuffle

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, MODELS_DIR
from src.datasets.commons import preprocessing
from src.samplers.hypercube_fingerprint_sampler import HypercubeFingerprintSampler


def make_sampled_embeddings(sequence_file_lines):
    with tf.device("/cpu:0"):
        encoder = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "autoencoders", "661k_conv_small_loc_pres_ld10")).encoder

    # todo: calculate split values on fraction of dataset, like in make_embeddings.py, but move the code to a new script
    split_values = np.array([[2.33112024], [3.2449895], [0.97849213], [2.04137627], [1.55365889],
                             [1.15269995], [1.27782217], [1.47252198], [1.28234643], [3.08180996]])
    sampler = HypercubeFingerprintSampler(encoder.output.shape[-1], split_values,
                                          name="hypercube-fingerprint-median-sampler")

    sampled = []
    for i, line in enumerate(sequence_file_lines):
        organism_id, sequence = line.strip().split(",")
        int_encoded = preprocessing.integer_encode(sequence)
        one_hot_encoded = preprocessing.one_hot_encode_np(int_encoded)
        windows = preprocessing.split_into_windows_np(one_hot_encoded, 128, 3)
        with tf.device("/cpu:0"):
            embeddings = encoder.predict(windows, batch_size=4096)
        sampled.append((organism_id, sampler.sample_np(embeddings)))
        if i % 1000 == 0:
            print(f"{multiprocessing.current_process().name}: {i}/{len(sequence_file_lines)}", flush=True)
    return sampled


if __name__ == "__main__":
    # preparation
    _, dataset_name, autoencoder_name, sampler_name, n_workers_str = sys.argv
    n_workers = int(n_workers_str)

    os.makedirs(os.path.join(DATA_DIR, dataset_name, "sampled-embeddings"))
    subsets = ["train", "test", "val"]

    for subset in subsets:
        with open(os.path.join(DATA_DIR, dataset_name, "fasta", f"{subset}.csv"), "rt") as in_file:
            # prepare data
            sequence_entries = in_file.readlines()
            n_sequences = len(sequence_entries)
            shuffle(sequence_entries)
            chunk_size = math.ceil(len(sequence_entries) / n_workers)
            chunks = [sequence_entries[i:i + chunk_size] for i in range(0, len(sequence_entries), chunk_size)]
            del sequence_entries

            # call workers
            pool = multiprocessing.Pool(n_workers)
            results = pool.imap(make_sampled_embeddings, chunks)
            pool.close()
            pool.join()

            with open(os.path.join(DATA_DIR, dataset_name, "sampled-embeddings", f"{subset}.csv"), "wt") as out_file:
                for chunk in tqdm(results, total=n_sequences, desc="Writing sampled embeddings"):
                    for organism_id, sampled_embeddings in chunk:
                        out_file.write(f"{organism_id};{json.dumps(sampled_embeddings.tolist())}\n")
