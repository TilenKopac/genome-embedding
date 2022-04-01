import math
import os
import pickle
import random
import zlib
from enum import Enum

import tensorflow as tf
from tqdm import tqdm

from src.datasets.commons import input_output, preprocessing


class TaxonomyDataset:

    def __init__(self, data_dir, encoder_name, sampler_name, data_subset, batch_size, tax_rank, limit=None):
        super(TaxonomyDataset, self).__init__()
        self.embeddings_dir = os.path.join(data_dir, "embeddings", encoder_name, sampler_name, data_subset)
        self.tax_recs_dir = os.path.join(data_dir, "taxonomy")
        try:
            with open(os.path.join(data_dir, "taxa_index.pkl"), "rb") as file:
                self.taxa_index = pickle.load(file)
        except FileNotFoundError:
            raise Exception(f"Taxa index not found on path {os.path.join(data_dir, 'taxa_index')}")
        self.n_labels = max(self.taxa_index[tax_rank.value].values())

        self.batch_size = batch_size
        self.tax_rank = tax_rank
        self.limit = limit
        if limit:
            self.n_batches = limit
        else:
            self.n_batches = self.calculate_n_batches()
        self.tf_dataset = self.instantiate_dataset()

    def get_record(self):
        filenames = list(set(os.listdir(self.embeddings_dir)).intersection(set(os.listdir(self.tax_recs_dir))))
        random.shuffle(filenames)
        for filename in filenames:
            try:
                embeddings = input_output.read_from_disk(self.embeddings_dir, filename)
                tax_rec = input_output.read_from_disk(self.tax_recs_dir, filename)
                if not tax_rec.get(self.tax_rank.value):
                    continue
            except zlib.error:
                continue
            tax_encoded = self.taxa_index[self.tax_rank.value][tax_rec[self.tax_rank.value]]

            yield tf.constant(tax_encoded, dtype=tf.int32), tf.constant(embeddings, dtype=tf.float32)

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.get_record, output_types=(tf.int32, tf.float32))
        dataset = dataset.map(lambda tax_id, embeddings: (
            preprocessing.one_hot_encode(tf.repeat(tax_id, [tf.shape(embeddings)[0]]), self.n_labels),
            embeddings
        ), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(
            lambda organism_ids, embeddings: tf.data.Dataset.from_tensor_slices((organism_ids, embeddings)))
        dataset = dataset.batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        return dataset

    def calculate_n_batches(self):
        # todo: change this calculation, as the number of windows does not directly determine the number of batches
        #  in this case because of samplers
        n_windows = 0
        filenames = set(os.listdir(self.embeddings_dir)).intersection(set(os.listdir(self.tax_recs_dir)))
        for filename in tqdm(filenames, desc="Calculating dataset size"):
            try:
                embeddings = input_output.read_from_disk(self.embeddings_dir, filename)
                tax_rec = input_output.read_from_disk(self.tax_recs_dir, filename)
                if tax_rec.get(self.tax_rank.value):
                    # n_windows += (len(nuc_rec.seq) - self.window_size) // self.step_size + 1
                    n_windows += 1
            except zlib.error:
                continue
        return math.floor(n_windows / self.batch_size)


class TaxonomicRankEnum(Enum):
    KINGDOM = "kingdom"
    GENUS = "genus"
    SPECIES = "species"
