import os.path
import zlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, MODELS_DIR
from src.datasets.commons import input_output, preprocessing
from src.samplers.centroid_sampler import CentroidSampler
from src.samplers.convex_hull_sampler import ConvexHullSampler
from src.samplers.hypercube_fingerprint_sampler import HypercubeFingerprintSampler
from src.samplers.random_sampler import RandomSampler
from src.samplers.sampler_enum import SamplerEnum


def embed_sequence(sequence):
    # integer encode
    int_encoded = preprocessing.integer_encode(sequence)

    # one-hot encode
    oh_encoded = preprocessing.one_hot_encode_np(int_encoded, depth=4)

    # split into windows
    windows = preprocessing.split_into_windows_np(oh_encoded, 128, 4)

    # embed
    embeddings = []
    batch_size = 4096
    for i in range(0, windows.shape[0], batch_size):
        inputs = tf.convert_to_tensor(windows[i:i + batch_size])
        preds = encoder(inputs).numpy()
        embeddings.append(preds)

    return np.concatenate(embeddings)


# encoder
autoencoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoders", "virus-conv-small-ld4"))
encoder = autoencoder.encoder
latent_dim = encoder.layers[-1].output_shape[-1]

# sampler
sampler_type = SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN

records_dir = os.path.join(DATA_DIR, "viruses/fasta")
output_dir = os.path.join(DATA_DIR, "viruses/embeddings", autoencoder.name,
                          sampler_type.value)
if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
    raise Exception(f"Directory '{output_dir}' already exists and is not empty")
else:
    os.makedirs(output_dir, exist_ok=True)

if sampler_type == SamplerEnum.CENTROID:
    sampler = CentroidSampler(name=SamplerEnum.CENTROID.value)
elif sampler_type == SamplerEnum.RANDOM:
    sampler = RandomSampler(n_points=100, name=SamplerEnum.RANDOM.value)
elif sampler_type == SamplerEnum.CONVEX_HULL:
    sampler = ConvexHullSampler(name=SamplerEnum.CONVEX_HULL.value)
elif sampler_type in [SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN, SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN_NORMALIZED,
                      SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE, SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE_NORMALIZED,
                      SamplerEnum.HYPERCUBE_FINGERPRINT_OCTILE, SamplerEnum.HYPERCUBE_FINGERPRINT_OCTILE_NORMALIZED]:
    # calculate split values (median or quartiles) of each dimension of training set embeddings
    if sampler_type in [SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN, SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN_NORMALIZED]:
        splits = 1
    elif sampler_type in [SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE,
                          SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE_NORMALIZED]:
        splits = 3
    elif sampler_type in [SamplerEnum.HYPERCUBE_FINGERPRINT_OCTILE, SamplerEnum.HYPERCUBE_FINGERPRINT_OCTILE_NORMALIZED]:
        splits = 7
    split_values = np.zeros((latent_dim, splits))
    percentiles = np.linspace(0, 100, splits + 2)[1:-1]
    norm = 0
    records_subdir = os.path.join(records_dir, "train")
    for filename in tqdm(os.listdir(records_subdir), desc="Calculating split values"):
        try:
            # load record
            record = input_output.read_from_disk(records_subdir, filename)
            if not preprocessing.is_sequence_valid(record.seq):
                continue
        except zlib.error:
            continue

        embeddings = embed_sequence(record.seq)

        # calculate split values
        split_values += np.percentile(embeddings, percentiles, axis=0).T
        # split_values += np.percentile(embeddings, percentiles, axis=0)
        # if sampler_type == SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN:
        #     split_values += np.median(embeddings, axis=0)[:, np.newaxis]
        # elif sampler_type == SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE:
        #     split_values += np.percentile(embeddings, [25, 50, 75], axis=0).T
        norm += 1

    # normalize split values
    split_values /= norm
    print(f"\nSplit values:\n{split_values}")

    # init sampler
    if sampler_type == SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN_NORMALIZED or sampler_type == SamplerEnum.HYPERCUBE_FINGERPRINT_QUARTILE_NORMALIZED:
        sampler = HypercubeFingerprintSampler(latent_dim, split_values, normalize=True, name=sampler_type.value)
    else:
        sampler = HypercubeFingerprintSampler(latent_dim, split_values, name=sampler_type.value)
else:
    raise Exception("Specified sampler type is not known")

for foldername in os.listdir(records_dir):
    records_subdir = os.path.join(records_dir, foldername)
    output_subdir = os.path.join(output_dir, foldername)
    os.makedirs(output_subdir)
    for filename in tqdm(os.listdir(records_subdir), desc="Creating sampled embeddings"):
        try:
            # load record
            record = input_output.read_from_disk(records_subdir, filename)
            if not preprocessing.is_sequence_valid(record.seq):
                continue
        except zlib.error:
            continue

        embeddings = embed_sequence(record.seq)

        # sample
        sampled = sampler.sample_np(embeddings)

        # save sampled embeddings
        input_output.save_to_disk(output_subdir, filename, sampled)
