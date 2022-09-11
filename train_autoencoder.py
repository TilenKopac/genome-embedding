import os
import time

import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, CHECKPOINTS_DIR, TRAIN_LOGS_DIR
from src.autoencoders.convolutional_small_autoencoder import ConvolutionalSmallAutoencoder
from src.datasets.genome_window_dataset import GenomeWindowDataset
from src.metrics.locality_preserving_loss import LocalityPreservingLoss
from src.metrics.reconstruction_accuracy import ReconstructionAccuracy

# dataset parameters
data_dir = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced_small")
window_size = 100
step_size = 3
batch_size = 4096
n_mutations = 1

train_dataset = GenomeWindowDataset(data_dir, "train", window_size, step_size, batch_size, n_mutations, shuffle=False)
val_dataset = GenomeWindowDataset(data_dir, "val", window_size, step_size, batch_size, n_mutations, shuffle=False)

# autoencoder parameters
latent_dim = 10
pool_size = 2
autoencoder_name = "661k_conv_small_loc_pres_ld10_ws100"

# training parameters
max_seq_weight = 1e-2
max_sim_weight = 1e-4
# max_seq_weight = 0.0
# max_sim_weight = 0.0
n_seq_windows = 3
seq_window_weights = tf.constant([1.0, 0.75, 0.5])
learning_rate = 1e-4
n_epochs = 10
# annealing
n_weight_cycles = 5
weight_cycle_proportion = 0.8

autoencoder = ConvolutionalSmallAutoencoder(window_size, latent_dim, pool_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# metrics
loss_fn = LocalityPreservingLoss(n_mutations, train_dataset.n_batches, batch_size, max_sim_weight, max_seq_weight,
                                 n_seq_windows, seq_window_weights, n_weight_cycles, weight_cycle_proportion)
train_acc_metric = ReconstructionAccuracy()
val_acc_metric = ReconstructionAccuracy()


@tf.function
def train_step(iteration, organism_ids, originals):
    with tf.GradientTape() as tape:
        # encode windows and reconstruct input from latent encoding
        encoded = autoencoder.encoder(originals)
        reconstructed = autoencoder.decoder(encoded)

        # compute loss
        loss = loss_fn.compute_loss(iteration, organism_ids, originals, encoded, reconstructed)

    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    # compute accuracy
    train_acc_metric.compute_accuracy(originals, reconstructed)


@tf.function
def val_step(originals):
    encoded = autoencoder.encoder(originals)
    reconstructed = autoencoder.decoder(encoded)

    val_acc_metric.compute_accuracy(originals, reconstructed)


# tensorboard training logs
train_log_dir = os.path.join(TRAIN_LOGS_DIR, "autoencoders", autoencoder_name)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# model checkpoints
checkpoints_path = os.path.join(CHECKPOINTS_DIR, "autoencoders", autoencoder_name)
checkpoint = tf.train.Checkpoint(autoencoder=autoencoder, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoints_path, max_to_keep=None)

# if a checkpoint exists, restore the latest one
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Latest checkpoint restored!")

for epoch in range(n_epochs):
    start = time.time()

    step_counter = tf.constant(0.0)

    # training loop
    for i, (organism_ids, originals) in tqdm(enumerate(train_dataset.tf_dataset), total=train_dataset.n_batches,
                                             desc=f"Epoch {epoch} training"):
        train_step(step_counter, organism_ids, originals)
        step_counter += 1.0

        if i % 100 == 0:
            with train_summary_writer.as_default():
                iter = epoch * train_dataset.n_batches + i
                total_loss, rec_loss, seq_loss, sim_loss = loss_fn.results()
                train_acc = train_acc_metric.result()
                tf.summary.scalar(f"train_loss_iters", total_loss, step=iter)
                tf.summary.scalar(f"train_reconstruction_loss_iters", rec_loss, step=iter)
                tf.summary.scalar(f"train_sequentiality_loss_iters", seq_loss, step=iter)
                tf.summary.scalar(f"train_similarity_loss_iters", sim_loss, step=iter)
                tf.summary.scalar(f"train_accuracy_iters", train_acc, step=iter)

    # validation loop
    for _, originals in tqdm(val_dataset.tf_dataset, total=val_dataset.n_batches,
                             desc=f"Epoch {epoch} validation"):
        val_step(originals)

    if (epoch + 1) % 1 == 0:
        checkpoint_path = checkpoint_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {checkpoint_path}")

    total_loss, rec_loss, seq_loss, sim_loss = loss_fn.results()
    loss_fn.reset()
    train_acc = train_acc_metric.result()
    train_acc_metric.reset()
    val_acc = val_acc_metric.result()
    val_acc_metric.reset()
    print(f"Epoch {epoch + 1} Reconstruction loss {rec_loss:.4e} "
          f"Sequentiality loss {seq_loss:.4e} Similarity loss {sim_loss:.4e} "
          f"Train accuracy {train_acc * 100:.2f} Validation accuracy {val_acc * 100:.2f}")

    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss_epochs", total_loss, step=epoch)
        tf.summary.scalar(f"train_reconstruction_loss_epochs", rec_loss, step=epoch)
        tf.summary.scalar(f"train_sequentiality_loss_epochs", seq_loss, step=epoch)
        tf.summary.scalar(f"train_similarity_loss_epochs", sim_loss, step=epoch)
        tf.summary.scalar(f"train_accuracy_epochs", train_acc, step=epoch)
        tf.summary.scalar(f"val_accuracy_epochs", train_acc, step=epoch)

    print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")
