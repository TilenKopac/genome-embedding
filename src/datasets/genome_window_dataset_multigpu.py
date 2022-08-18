import os

import tensorflow as tf

from src.datasets.commons import preprocessing


class DistributedGenomeWindowDataset:

    def __init__(self, data_dir, data_subset, window_size, step_size, global_batch_size, n_mutations,
                 shuffle=False, limit=None):
        super(DistributedGenomeWindowDataset).__init__()
        self.nuc_recs_file = open(os.path.join(data_dir, "fasta", f"{data_subset}.csv"))
        self.subset = data_subset
        self.window_size = window_size
        self.step_size = step_size
        self.global_batch_size = global_batch_size
        self.n_mutations = n_mutations
        self.shuffle = shuffle
        self.limit = limit
        if limit:
            self.global_n_batches = limit
        else:
            self.global_n_batches = self.calculate_n_batches()

    def get_record(self):
        for index, line in enumerate(self.nuc_recs_file):
            _, sequence = line.strip().split(",")
            if not preprocessing.is_sequence_valid(sequence):
                continue

            encoded = preprocessing.integer_encode(sequence)
            yield tf.constant(index, dtype=tf.float32), tf.constant(encoded, dtype=tf.uint8)

    def instantiate_dataset(self, global_batch_size, input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.float32, tf.uint8))
        dataset = dataset.map(
            lambda organism_id, sequence: preprocessing.split_into_windows_with_id(organism_id, sequence,
                                                                                   self.window_size, self.step_size),
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
        if self.shuffle:
            dataset = dataset.shuffle(int(self.global_n_batches / (global_batch_size / batch_size) * 0.05))
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
        return n_windows * (1 + self.n_mutations) // self.global_batch_size
