import math
import os
import random
import zlib

import tensorflow as tf
from tqdm import tqdm

from src.datasets.commons import input_output, preprocessing


class GenomeWindowDataset:

    def __init__(self, data_dir, data_subset, window_size, step_size, batch_size, n_mutations, limit=None):
        super(GenomeWindowDataset).__init__()
        self.nuc_recs_dir = os.path.join(data_dir, "fasta", data_subset)
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
            try:
                record = input_output.read_from_disk(self.nuc_recs_dir, filename)
                if not preprocessing.is_sequence_valid(record.seq):
                    continue
            except zlib.error:
                continue

            encoded = preprocessing.integer_encode(record.seq)
            yield tf.constant(index, dtype=tf.float32), tf.constant(encoded, dtype=tf.uint8)

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.float32, tf.uint8))
        dataset = dataset.map(
            lambda organism_id, sequence: preprocessing.split_into_windows(organism_id, sequence, self.window_size,
                                                                           self.step_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(
            lambda organism_ids, windows: tf.data.Dataset.from_tensor_slices((organism_ids, windows)))
        dataset = dataset.batch(self.batch_size // (self.n_mutations + 1), drop_remainder=True,
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda organism_ids, windows: (organism_ids, preprocessing.one_hot_encode_sequences(windows)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if self.n_mutations > 0:
            dataset = dataset.map(lambda organism_ids, windows: (
                preprocessing.make_mutations(organism_ids, windows, self.n_mutations, self.batch_size, self.window_size)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        return dataset

    def calculate_n_batches(self):
        n_windows = 0
        filenames = os.listdir(self.nuc_recs_dir)
        for filename in tqdm(filenames, desc="Calculating dataset size"):
            try:
                record = input_output.read_from_disk(self.nuc_recs_dir, filename)
                if preprocessing.is_sequence_valid(record.seq):
                    n_windows += (len(record.seq) - self.window_size) // self.step_size + 1
            except zlib.error:
                continue
        return math.floor(n_windows / (self.batch_size / (self.n_mutations + 1)))
