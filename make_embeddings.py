import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, MODELS_DIR
from src.datasets.genome_window_dataset import GenomeWindowDataset
# dataset parameters
from src.samplers.hypercube_fingerprint_sampler import HypercubeFingerprintSampler

dataset_path = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced")
window_size = 128
step_size = 3
batch_size = 4096
n_mutations = 0

subsets = ["train", "test", "val"]
datasets = [GenomeWindowDataset(dataset_path, subset, window_size, step_size, batch_size, n_mutations, shuffle=False)
            for subset in subsets]

# (auto)encoder
autoencoder_name = "661k_conv_small_loc_pres_ld10"
encoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoders", autoencoder_name)).encoder

# sampler
# todo: parametrize sampler initialization (like in "sample_embeddings.py")
split_value_dataset = GenomeWindowDataset(dataset_path, "val", window_size, step_size, batch_size, n_mutations,
                                          limit=10000)
splits = 1
split_values = np.zeros((encoder.output.shape[-1], splits))
percentiles = np.linspace(0, 100, splits + 2)[1:-1]
norm = 0
for _, originals in tqdm(split_value_dataset.tf_dataset, total=10000, desc=f"Calculating split values"):
    embeddings = encoder(originals)
    split_values += np.percentile(embeddings, percentiles, axis=0).T
    norm += 1
split_values /= norm
sampler = HypercubeFingerprintSampler(encoder.output.shape[-1], split_values, normalize=True,
                                      name="hypercube-fingerprint-median-sampler")

# create output directory
for dataset in datasets:
    os.makedirs(os.path.join(dataset_path, "embeddings", autoencoder_name), exist_ok=True)

ids_to_write = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
embeddings_to_write = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
for dataset in datasets:
    # output_dir = os.path.join(dataset_path, "embeddings", autoencoder_name, dataset.subset)
    output_file = open(os.path.join(dataset_path, "embeddings", autoencoder_name, f"{dataset.subset}.csv"), "wt")
    arr_write_index = 0
    remaining_ids = None
    remaining_embeddings = None

    for organism_indices, originals in tqdm(dataset.tf_dataset, total=dataset.n_batches,
                                            desc=f"Embedding {dataset.subset} dataset"):
        # embed sequences
        embeddings = encoder(originals)

        # store organism ids and sequences in arrays
        ids_to_write = ids_to_write.write(arr_write_index, organism_indices)
        embeddings_to_write = embeddings_to_write.write(arr_write_index, embeddings)
        arr_write_index += 1

        # periodically write embeddings to disk
        if arr_write_index == 1000:
            if remaining_ids is not None and remaining_embeddings is not None:
                indices_to_write_concat = tf.concat([remaining_ids, ids_to_write.concat()], axis=0)
                embeddings_to_write_concat = tf.concat([remaining_embeddings, embeddings_to_write.concat()], axis=0)
            else:
                indices_to_write_concat = ids_to_write.concat()
                embeddings_to_write_concat = embeddings_to_write.concat()

            for organism_index in tf.unique(indices_to_write_concat).y[:-1]:
                # get all embeddings of the organism with organism_id
                organism_embeddings = tf.boolean_mask(embeddings_to_write_concat,
                                                      indices_to_write_concat == organism_index)

                # sample embeddings
                sampled_embeddings = sampler.sample_np(organism_embeddings.numpy())

                # dump sampled embeddings with their corresponding organism id as lines in the dataset csv file
                output_file.write(f"{dataset.index_org_id_map[int(organism_index)]};"
                                  f"{json.dumps(sampled_embeddings.tolist())}\n")

                # dump embeddings as a numpy array of shape [n_embeddings, latent_dim]
                # with gzip.open(os.path.join(output_dir, dataset.id_filename_map[int(organism_id)]), "wb") as file:
                #     pickle.dump(organism_embeddings.numpy(), file)

            # carry last organism's id and embeddings to next loop, since they usually get cut off between loops
            remaining_ids = tf.boolean_mask(indices_to_write_concat, indices_to_write_concat == organism_indices[-1])
            remaining_embeddings = tf.boolean_mask(embeddings_to_write_concat,
                                                   indices_to_write_concat == organism_indices[-1])

            # reset write index, so that id and embedding arrays get overwritten from the start
            arr_write_index = 0

    output_file.close()
