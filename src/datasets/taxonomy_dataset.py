import json
import os
import pickle
from collections import Counter
from enum import Enum

import tensorflow as tf

from definitions import DATA_DIR
from src.samplers.sampler_enum import SamplerEnum


class TaxonomyDataset:

    def __init__(self, data_dir, subset, encoder_name, sampler_name, batch_size, taxa_index, organism_taxa,
                 tax_rank, shuffle=False, limit=None):
        super(TaxonomyDataset, self).__init__()
        self.subset = subset
        self.taxa_index = taxa_index
        self.organism_taxa = organism_taxa
        self.n_labels = max(self.taxa_index[tax_rank.value].values())
        self.tax_rank = tax_rank
        self.data = self.prepare_data(os.path.join(data_dir, "embeddings", encoder_name, sampler_name, f"{subset}.csv"))
        self.class_weights = self.calculate_class_weights()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.limit = limit
        if limit:
            self.n_batches = limit
        else:
            self.n_batches = len(self.data["labels"]) // batch_size
        self.tf_dataset = self.instantiate_dataset()

    def prepare_data(self, nuc_recs_file_path):
        data = {"labels": [], "embeddings": []}
        with open(nuc_recs_file_path, "rt") as nuc_recs_file:
            for i, line in enumerate(nuc_recs_file):
                # organism id is obtained from the first part of the first field (the second part is the sequence id)
                organism_id = line.split(";")[0].split(".")[0]
                # the embedding is obtained from the second field
                try:
                    embedding = json.loads(line.split(";")[1])
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

                data["labels"].append(tax_encoded)
                data["embeddings"].append(embedding)

                # if i == 1000 * 4096:
                #     break

        return data

    def calculate_class_weights(self):
        # count examples of all classes
        class_counter = Counter(self.data["labels"])
        total = sum(class_counter.values())

        # prepare weights
        classes = []
        weights = []
        for cls, count in class_counter.items():
            classes.append(cls)
            weights.append(1 - count / total)

        # normalize weights
        norm_factor = sum(weights)
        for weight in weights:
            weight /= norm_factor

        # create lookup table
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(classes), tf.constant(weights)),
            default_value=1.0
        )

        return table

    def instantiate_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(self.data["labels"]),
                                                      tf.constant(self.data["embeddings"])))
        dataset = dataset.map(lambda tax_id, embedding: (tf.one_hot(tax_id, self.n_labels), embedding,
                                                         self.class_weights.lookup(tax_id)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(int(self.n_batches * 0.1 * self.batch_size))
        dataset = dataset.batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        if self.limit:
            dataset = dataset.take(self.limit)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset


class TaxonomicRankEnum(Enum):
    KINGDOM = "kingdom"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"
