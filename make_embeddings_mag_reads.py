import json
import os
import pickle
import sys

import tensorflow as tf
from Bio import SeqIO
from tqdm import tqdm

from definitions import DATA_DIR, MODELS_DIR
from src.datasets.commons import preprocessing

# read arguments
from src.samplers.sampler_enum import SamplerEnum

_, autoencoder_name, process_index = sys.argv
process_index = int(process_index)

# prepare encoder
encoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoders", autoencoder_name)).encoder
window_size = encoder.input.shape[1]

dataset_dir = os.path.join(DATA_DIR, "deepmicrobes_mag_reads")
input_dir = os.path.join(dataset_dir, "fasta")
output_dir = os.path.join(dataset_dir, "embeddings", autoencoder_name, SamplerEnum.NO_SAMPLER.value)
# create target dir
try:
    os.makedirs(output_dir)
except FileExistsError:
    # another process already created the output directory
    pass

# prepare dictionary of organisms' taxa
with open(os.path.join(dataset_dir, "organism_taxa.pkl"), "rb") as file:
    organism_taxa = pickle.load(file)

# each process prepares embeddings for 7 files
for in_filename in tqdm(os.listdir(input_dir)[process_index * 7:process_index * 7 + 7]):
    org_id = in_filename.split(".")[0].split("_")[1]
    if org_id in organism_taxa:
        out_file_path = os.path.join(output_dir, f"{org_id}.csv")
        with open(os.path.join(dataset_dir, "fasta", in_filename), "rt") as in_file, open(out_file_path, "wt") as out_file:
            records = SeqIO.parse(in_file, "fasta")
            encoded_reads = []
            for record in records:
                # one hot encode
                oh_encoded = [preprocessing.nuc_onehot_enc[nuc] for nuc in record]

                # cut read so that it fits into our encoder
                encoded_reads.append(oh_encoded[:window_size])

                # skip the next record, which is just a reverse complement of the current one
                next(records)

            # embed all organism's reads using our (auto)encoder
            embeddings = encoder(tf.constant(encoded_reads, dtype=tf.float32))

            # store embeddings
            for embedding in embeddings:
                out_file.write(f"{org_id};{json.dumps(embedding.numpy().tolist())}\n")
