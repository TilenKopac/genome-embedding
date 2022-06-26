import gzip
import os.path
import pickle
import zlib

import numpy as np
from tqdm import tqdm

from definitions import DATA_DIR
from src.samplers.centroid_sampler import CentroidSampler
from src.samplers.convex_hull_sampler import ConvexHullSampler
from src.samplers.hypercube_fingerprint_sampler import HypercubeFingerprintSampler
from src.samplers.random_sampler import RandomSampler
from src.samplers.sampler_enum import SamplerEnum

# autoencoder
autoencoder_name = "661k_conv_small_loc_pres_ld10"
latent_dim = 10

# sampler
sampler_type = SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN

# dataset
dataset_path = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced")
embeddings_dir = os.path.join(dataset_path, "embeddings", autoencoder_name)
samples_dir = os.path.join(dataset_path, "samples", autoencoder_name, sampler_type.value)

# output_dir = os.path.join(DATA_DIR, "viruses/embeddings", autoencoder.name, sampler_type.value)

# if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
#     raise Exception(f"Directory '{output_dir}' already exists and is not empty")
# else:
#     os.makedirs(output_dir, exist_ok=True)

for subdir in os.listdir(embeddings_dir):
    os.makedirs(os.path.join(samples_dir, subdir))

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
    embeddings_subdir = os.path.join(embeddings_dir, "val")
    for filename in tqdm(os.listdir(embeddings_subdir), desc="Calculating split values"):
        try:
            # load record
            with gzip.open(os.path.join(embeddings_subdir, filename)) as file:
                embeddings = pickle.load(file)
        except (zlib.error, EOFError):
            continue

        # calculate split values
        split_values += np.percentile(embeddings, percentiles, axis=0).T
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

for subdir in os.listdir(embeddings_dir):
    embeddings_subdir = os.path.join(embeddings_dir, subdir)
    output_subdir = os.path.join(samples_dir, subdir)
    for filename in tqdm(os.listdir(embeddings_subdir), desc=f"Sampling {subdir} embeddings"):
        try:
            # load record
            with gzip.open(os.path.join(embeddings_subdir, filename)) as file:
                embeddings = pickle.load(file)
        except (zlib.error, EOFError):
            continue

        # sample
        sampled = sampler.sample_np(embeddings)

        # save sampled embeddings
        with gzip.open(os.path.join(output_subdir, filename), "wb") as file:
            pickle.dump(sampled, file)
