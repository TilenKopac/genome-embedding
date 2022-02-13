import gzip
import math
import os
import pickle

import tensorflow as tf
from tqdm import tqdm


class Dataset:
    nuc_enc = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "G": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.25, 0.25, 0.25, 0.25]}
    nuc_mutations = tf.constant(list(nuc_enc.values()), dtype=tf.float16)

    def __init__(self, data_dir, window_size, step_size, batch_size, n_mutations):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_mutations = n_mutations
        if batch_size % (n_mutations + 1) != 0:
            raise ValueError("Batch size has to be divisible by n_mutations + 1!")
        self.tf_dataset = self.instantiate_dataset()
        self.n_batches = self.calculate_n_batches()

    def get_record(self):
        for record_id, filename in enumerate(os.listdir(self.data_dir)):
            with gzip.open(f"{self.data_dir}/{filename}", "rb") as file:
                record = pickle.load(file)
                encoded = []
                for nuc in record.seq:
                    enc = self.nuc_enc.get(nuc)
                    if enc:
                        encoded.append(enc)
                    else:
                        continue
            yield tf.constant(record_id, dtype=tf.int32), tf.constant(encoded, dtype=tf.float16)

    @tf.function
    def mutate_nucleotide(self, nucleotide):
        if tf.reduce_all(nucleotide - tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float16) == 0):
            weights = [0.95, 0.03, 0.01, 0.01]
        elif tf.reduce_all(nucleotide - tf.constant([0.0, 1.0, 0.0, 0.0], dtype=tf.float16) == 0):
            weights = [0.03, 0.95, 0.01, 0.01]
        elif tf.reduce_all(nucleotide - tf.constant([0.0, 0.0, 1.0, 0.0], dtype=tf.float16) == 0):
            weights = [0.01, 0.01, 0.95, 0.03]
        elif tf.reduce_all(nucleotide - tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float16) == 0):
            weights = [0.01, 0.01, 0.03, 0.95]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]

        ind = tf.random.categorical(tf.math.log([weights]), 1, dtype=tf.int32)[0][0]
        return self.nuc_mutations[ind]

    @tf.function
    def make_mutations(self, sequence):
        copies = tf.tile(sequence, [self.n_mutations, 1])
        mutated = tf.reshape(tf.map_fn(self.mutate_nucleotide, copies), [self.n_mutations, tf.shape(sequence)[0], 4])
        return tf.concat([[sequence], mutated], axis=0)

    @tf.function
    def split_into_windows(self, sequences):
        n_windows = (tf.shape(sequences[0])[0] - self.window_size) // self.step_size + 1
        indices = tf.reshape(tf.tile(tf.range(0, self.window_size), [n_windows]), (n_windows, self.window_size))
        increment = tf.cast(tf.linspace([0] * self.window_size, [(n_windows - 1) * self.step_size] * self.window_size, n_windows), tf.int32)
        indices += increment
        windows = [tf.gather(sequences[i], indices) for i in range(self.n_mutations + 1)]
        interleave_ind = [tf.range(0, tf.shape(windows[0])[0] * (self.n_mutations + 1), self.n_mutations + 1) + i
                          for i in range(self.n_mutations + 1)]
        interleaved = tf.dynamic_stitch(interleave_ind, windows)
        return interleaved

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.int32, tf.float16))
        dataset = dataset.map(lambda organism_id, sequence: (organism_id, self.make_mutations(sequence)), num_parallel_calls=10)
        dataset = dataset.map(lambda organism_id, sequences: (organism_id, self.split_into_windows(sequences)), num_parallel_calls=10)
        dataset = dataset.flat_map(lambda organism_id, sequences:
                                   tf.data.Dataset.from_tensor_slices((tf.repeat(organism_id, tf.shape(sequences)[0]), sequences)))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(1000, seed=2022)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def calculate_n_batches(self):
        n_windows = 0
        for filename in tqdm(os.listdir(self.data_dir), desc="Calculating dataset size"):
            with gzip.open(f"{self.data_dir}/{filename}", "rb") as file:
                record = pickle.load(file)
                n_windows += (len(record.seq) - self.window_size) // self.step_size + 1
        return math.ceil(n_windows / (self.batch_size / (self.n_mutations + 1)))
