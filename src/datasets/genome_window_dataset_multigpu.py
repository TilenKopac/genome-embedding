import gzip
import math
import os
import pickle
import zlib
from random import Random

import tensorflow as tf
from tqdm import tqdm

from src.datasets.commons import preprocessing


class DistributedGenomeWindowDataset:

    def __init__(self, data_dir, data_subset, window_size, step_size, n_mutations, shuffle_buffer_size=None,
                 limit=None):
        super(DistributedGenomeWindowDataset).__init__()
        self.nuc_recs_dir = os.path.join(data_dir, "fasta", data_subset)
        self.window_size = window_size
        self.step_size = step_size
        self.n_mutations = n_mutations
        self.shuffle_buffer_size = shuffle_buffer_size
        self.limit = limit
        if limit:
            self.n_batches = limit
        # else:
        #     self.n_batches = self.calculate_n_batches(global_batch_size)
        self.filenames = sorted(os.listdir(self.nuc_recs_dir))
        Random(2022).shuffle(self.filenames)

    def get_record(self):
        for index, filename in enumerate(self.filenames):
            try:
                with gzip.open(os.path.join(self.nuc_recs_dir, filename)) as file:
                    record = pickle.load(file)
                if not preprocessing.is_sequence_valid(record.seq):
                    continue
            except zlib.error:
                continue

            encoded = preprocessing.integer_encode(record.seq)
            yield tf.constant(index, dtype=tf.float32), tf.constant(encoded, dtype=tf.uint8)

    def instantiate_dataset(self, global_batch_size, input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.float32, tf.uint8))
        dataset = dataset.map(
            lambda organism_id, sequence: preprocessing.split_into_windows(organism_id, sequence, self.window_size,
                                                                           self.step_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(
            lambda organism_ids, windows: tf.data.Dataset.from_tensor_slices((organism_ids, windows)))
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size // (self.n_mutations + 1), drop_remainder=True,
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda organism_ids, windows: (organism_ids, preprocessing.one_hot_encode_sequences(windows)),
            num_parallel_calls=tf.data.AUTOTUNE)
        if self.n_mutations > 0:
            dataset = dataset.map(lambda organism_ids, windows: (
                preprocessing.make_mutations(organism_ids, windows, self.n_mutations, batch_size,
                                             self.window_size)), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        return dataset

    def calculate_n_batches(self, global_batch_size):
        n_windows = 0
        for entry in tqdm(os.scandir(self.nuc_recs_dir), desc="Calculating number of batches"):
            # todo: iterate over files only if number of nucleotides per sequence is not already present in some file
            #  (list of counts)
            try:
                with gzip.open(entry) as file:
                    record = pickle.load(file)
                if preprocessing.is_sequence_valid(record.seq):
                    n_windows += (len(record.seq) - self.window_size) // self.step_size + 1
            except zlib.error:
                continue
        print(math.floor(n_windows / (global_batch_size / (self.n_mutations + 1))))
        return math.floor(n_windows / (global_batch_size / (self.n_mutations + 1)))
