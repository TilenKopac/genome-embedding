import os

import tensorflow as tf

from definitions import MODELS_DIR, CHECKPOINTS_DIR
from src.autoencoders.convolutional_small_autoencoder import ConvolutionalSmallAutoencoder

# prepare architecture
autoencoder_name = "661k_conv_small_loc_pres_ld10_ws100"
window_size = 100
latent_dim = 10
pool_size = 2
autoencoder = ConvolutionalSmallAutoencoder(window_size, latent_dim, pool_size)
autoencoder(tf.random.uniform([1, window_size, 4]))

# load checkpoint
autoencoder_checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "autoencoders", autoencoder_name)
checkpoint = tf.train.Checkpoint(model=autoencoder)
latest_checkpoint = tf.train.latest_checkpoint(autoencoder_checkpoints_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    # save model
    tf.keras.models.save_model(autoencoder, os.path.join(MODELS_DIR, "autoencoders", autoencoder_name))
else:
    print("No checkpoints found for specified model")