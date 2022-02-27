import math
import os
import random

import tensorflow as tf
from tqdm import tqdm

from commons import nucleotide_encoding, input_output


class GenomeWindowDataset:
    # N is encoded as all zeros based on the work of Bartoszewicz (https://doi.org/10.1101/2020.01.29.925354)
    integer_onehot_enc = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "G": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0]}
    nuc_onehot_enc_list = tf.constant(list(integer_onehot_enc.values()), dtype=tf.float32)

    def __init__(self, data_dir, window_size, step_size, batch_size, n_mutations, limit=None):
        super(GenomeWindowDataset).__init__()
        self.nuc_recs_dir = os.path.join(data_dir, "fasta")
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_mutations = n_mutations
        if batch_size % (n_mutations + 1) != 0:
            raise ValueError("Batch size has to be divisible by n_mutations + 1!")
        self.limit = limit
        if limit:
            self.n_batches = limit
        else:
            self.n_batches = self.calculate_n_batches()
        self.tf_dataset = self.instantiate_dataset()

    def get_record(self):
        filenames = os.listdir(self.nuc_recs_dir)
        random.shuffle(filenames)
        for index, filename in enumerate(filenames):
            record = input_output.read_from_disk(self.nuc_recs_dir, filename)
            if not nucleotide_encoding.sequence_valid(record.seq):
                continue

            encoded = nucleotide_encoding.integer_encode(record.seq)
            yield tf.constant(index, dtype=tf.float32), tf.constant(encoded, dtype=tf.uint8)

    @tf.function
    def split_into_windows(self, organism_id, sequence):
        n_windows = (tf.shape(sequence)[0] - self.window_size) // self.step_size + 1
        organism_ids = tf.repeat(organism_id, n_windows)
        indices = tf.reshape(tf.tile(tf.range(0, self.window_size), [n_windows]), (n_windows, self.window_size))
        increment = tf.cast(
            tf.linspace([0] * self.window_size, [(n_windows - 1) * self.step_size] * self.window_size, n_windows),
            tf.int32)
        indices += increment
        windows = tf.gather(sequence, indices)
        return organism_ids, windows

    @tf.function
    def one_hot_encode(self, batch):
        encoded = tf.one_hot(batch, depth=len(nucleotide_encoding.nucleotides))
        return encoded

    @tf.function
    def mutate_nucleotide(self, nucleotide):
        if tf.reduce_all(nucleotide - self.nuc_onehot_enc_list[0] == 0):
            weights = [0.95, 0.03, 0.01, 0.01]
        elif tf.reduce_all(nucleotide - self.nuc_onehot_enc_list[1] == 0):
            weights = [0.03, 0.95, 0.01, 0.01]
        elif tf.reduce_all(nucleotide - self.nuc_onehot_enc_list[2] == 0):
            weights = [0.01, 0.01, 0.95, 0.03]
        elif tf.reduce_all(nucleotide - self.nuc_onehot_enc_list[3] == 0):
            weights = [0.01, 0.01, 0.03, 0.95]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]

        ind = tf.random.categorical(tf.math.log([weights]), 1, dtype=tf.int32)[0][0]
        return tf.one_hot(ind, depth=len(nucleotide_encoding.nucleotides))

    @tf.function
    def make_mutations(self, organism_ids, sequences):
        copies = tf.repeat(sequences, [self.n_mutations], axis=0)
        n_mut_windows = self.batch_size // (self.n_mutations + 1) * self.n_mutations
        mutated = tf.reshape(
            tf.vectorized_map(self.mutate_nucleotide, tf.reshape(copies, [n_mut_windows * self.window_size,
                                                                          len(nucleotide_encoding.nucleotides)])),
            [n_mut_windows, self.window_size, len(nucleotide_encoding.nucleotides)])

        # stitch originals and mutations into one tensor
        originals_ind = tf.range(0, self.batch_size, self.n_mutations + 1, tf.int32)
        mutations_ind = tf.reshape(
            tf.transpose(tf.stack([tf.range(i, self.batch_size, self.n_mutations + 1, tf.int32)
                                   for i in range(1, self.n_mutations + 1)])),
            [n_mut_windows])
        stitched = tf.dynamic_stitch([originals_ind, mutations_ind], [sequences, mutated])
        return tf.repeat(organism_ids, self.n_mutations + 1), stitched

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.float32, tf.uint8))
        dataset = dataset.map(lambda organism_id, sequence: self.split_into_windows(organism_id, sequence),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(
            lambda organism_ids, windows: tf.data.Dataset.from_tensor_slices((organism_ids, windows)))
        dataset = dataset.batch(self.batch_size // (self.n_mutations + 1), drop_remainder=True,
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda organism_ids, windows: (organism_ids, self.one_hot_encode(windows)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda organism_ids, windows: (self.make_mutations(organism_ids, windows)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        return dataset

    def calculate_n_batches(self):
        n_windows = 0
        filenames = os.listdir(self.nuc_recs_dir)
        for filename in tqdm(filenames, desc="Calculating dataset size"):
            record = input_output.read_from_disk(self.nuc_recs_dir, filename)
            if nucleotide_encoding.sequence_valid(record.seq):
                n_windows += (len(record.seq) - self.window_size) // self.step_size + 1
        return math.floor(n_windows / (self.batch_size / (self.n_mutations + 1)))
