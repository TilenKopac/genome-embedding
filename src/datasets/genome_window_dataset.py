import math
import os
import zlib

import tensorflow as tf
from tqdm import tqdm

from src.datasets.commons import input_output, preprocessing


class GenomeWindowDataset:

    def __init__(self, data_dir, data_subset, window_size, step_size, batch_size, n_mutations,
                 shuffle=False, limit=None):
        super(GenomeWindowDataset).__init__()
        self.nuc_recs_file = open(os.path.join(data_dir, "fasta", f"{data_subset}.csv"))
        self.index_org_id_map = {}
        self.subset = data_subset
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_mutations = n_mutations
        if batch_size % (n_mutations + 1) != 0:
            raise ValueError("Batch size has to be divisible by n_mutations + 1!")
        self.shuffle = shuffle
        self.limit = limit
        if limit:
            self.n_batches = limit
        else:
            self.n_batches = self.calculate_n_batches()
        self.tf_dataset = self.instantiate_dataset()

    def get_record(self):
        for index, line in enumerate(self.nuc_recs_file):
            org_id, sequence = line.strip().split(",")
            if not preprocessing.is_sequence_valid(sequence):
                continue

            self.index_org_id_map[index] = org_id
            encoded = preprocessing.integer_encode(sequence)
            yield tf.constant(index, dtype=tf.float32), tf.constant(encoded, dtype=tf.uint8)

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.float32, tf.uint8))
        dataset = dataset.map(lambda organism_id, sequence:
                              preprocessing.split_into_windows_with_id(organism_id, sequence,
                                                                       self.window_size, self.step_size),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(lambda organism_ids, windows:
                                   tf.data.Dataset.from_tensor_slices((organism_ids, windows)))
        dataset = dataset.batch(self.batch_size // (self.n_mutations + 1), drop_remainder=True,
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda organism_ids, windows:
                              (organism_ids, preprocessing.one_hot_encode_sequences(windows)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if self.n_mutations > 0:
            dataset = dataset.map(lambda organism_ids, windows:
                                  (preprocessing.make_mutations(organism_ids, windows, self.n_mutations,
                                                                self.batch_size, self.window_size)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        if self.shuffle:
            dataset = dataset.shuffle(int(self.n_batches * 0.05))
        return dataset

    def prepare_for_epoch(self):
        self.nuc_recs_file.seek(0)

    def calculate_n_batches(self):
        print(f"Calculating number of batches for {self.subset} dataset. This might take a while...")
        n_windows = 0
        for line in self.nuc_recs_file:
            _, seq = line.strip().split(",")
            if preprocessing.is_sequence_valid(seq):
                n_windows += (len(seq) - self.window_size) // self.step_size + 1
        self.prepare_for_epoch()
        return n_windows * (1 + self.n_mutations) // self.batch_size
