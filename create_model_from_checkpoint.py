import os
import pickle

import tensorflow as tf

from definitions import MODELS_DIR, CHECKPOINTS_DIR, DATA_DIR
from src.autoencoders.convolutional_small_autoencoder import ConvolutionalSmallAutoencoder
from src.classifiers.taxonomy_classifier import TaxonomyClassifier
from src.datasets.taxonomy_dataset import TaxonomyDataset, TaxonomicRankEnum
from src.samplers.sampler_enum import SamplerEnum

# # prepare architecture
# autoencoder_name = "661k_conv_small_loc_pres_ld10_ws100"
# window_size = 100
# latent_dim = 10
# pool_size = 2
# autoencoder = ConvolutionalSmallAutoencoder(window_size, latent_dim, pool_size)
# autoencoder(tf.random.uniform([1, window_size, 4]))
#
# # load checkpoint
# autoencoder_checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "autoencoders", autoencoder_name)
# checkpoint = tf.train.Checkpoint(model=autoencoder)
# latest_checkpoint = tf.train.latest_checkpoint(autoencoder_checkpoints_dir)
# if latest_checkpoint:
#     checkpoint.restore(latest_checkpoint)
#     # save model
#     tf.keras.models.save_model(autoencoder, os.path.join(MODELS_DIR, "autoencoders", autoencoder_name))
# else:
#     print("No checkpoints found for specified model")


dataset_dir = os.path.join(DATA_DIR, "deepmicrobes_mag_reads")
encoder_name = "661k_conv_small_elu_loc_pres_ld10_ws100_take_2"
sampler_name = SamplerEnum.NO_SAMPLER.value
window_size = 100
batch_size = 4096
tax_rank = TaxonomicRankEnum.GENUS
with open(os.path.join(dataset_dir, "taxa_index.pkl"), "rb") as file:
    taxa_index = pickle.load(file)
with open(os.path.join(dataset_dir, "organism_taxa.pkl"), "rb") as file:
    organism_taxa = pickle.load(file)
dataset = TaxonomyDataset(dataset_dir, "val", encoder_name, sampler_name, batch_size, taxa_index, organism_taxa,
                          tax_rank)

# prepare architecture
autoencoder_name = "661k_conv_small_elu_loc_pres_ld10_ws100_take_2"
classifier_name = "mag-reads-genus-classifier-4-layers-32-units"
window_size = 100
latent_dim = 10
pool_size = 2
classifier = TaxonomyClassifier(dataset.n_labels, n_layers=4, n_units=32)
classifier(next(iter(dataset.tf_dataset))[1])

# load checkpoint
classifiers_checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "classifiers", autoencoder_name,
                                           SamplerEnum.NO_SAMPLER.value, classifier_name)
checkpoint = tf.train.Checkpoint(model=classifier)
latest_checkpoint = tf.train.latest_checkpoint(classifiers_checkpoints_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    # save model
    tf.keras.models.save_model(classifier, os.path.join(MODELS_DIR, "temp", classifier_name))
else:
    print("No checkpoints found for specified model")
