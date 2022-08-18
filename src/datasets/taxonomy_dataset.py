import json
import os
from enum import Enum

import tensorflow as tf

from src.datasets.commons import preprocessing


class TaxonomyDataset:

    def __init__(self, data_dir, data_subset, encoder_name, sampler_name, batch_size, taxa_index, organism_taxa,
                 tax_rank, shuffle_buffer_size=None, limit=None):
        super(TaxonomyDataset, self).__init__()
        self.nuc_recs_file = open(os.path.join(data_dir, "sampled-embeddings", encoder_name, sampler_name,
                                               f"{data_subset}.csv"))
        self.taxa_index = taxa_index
        self.organism_taxa = organism_taxa
        self.n_labels = max(self.taxa_index[tax_rank.value].values())
        self.batch_size = batch_size
        self.tax_rank = tax_rank
        self.shuffle_buffer_size = shuffle_buffer_size
        self.limit = limit
        if limit:
            self.n_batches = limit
        self.tf_dataset = self.instantiate_dataset()

    def get_record(self):
        for line in self.nuc_recs_file:
            # organism id is obtained from the first part of the first field (the second part is the sequence id)
            organism_id = line.split(";")[0].split(".")[0]
            # embedding(s) is obtained from the second field
            try:
                embeddings = json.loads(line.split(";")[1])
            except json.decoder.JSONDecodeError:
                print(line)
                break

            if self.organism_taxa.get(organism_id):
                taxon = self.organism_taxa.get(organism_id)[self.tax_rank.value]
                if taxon:
                    tax_encoded = self.taxa_index[self.tax_rank.value][taxon]
                else:
                    # information about the specified taxonomic rank of the loaded record is not available
                    continue
            else:
                # information about the taxonomy of the loaded record is not available
                continue

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
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        return dataset

    def prepare_for_epoch(self):
        self.nuc_recs_file.seek(0)


class TaxonomicRankEnum(Enum):
    KINGDOM = "phylum"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"
