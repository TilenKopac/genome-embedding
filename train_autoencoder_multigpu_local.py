import os
import time

import tensorflow as tf
from tqdm import tqdm

# # split 1 physical GPU into 2 virtual GPUs
tf.config.set_logical_device_configuration(tf.config.list_physical_devices("GPU")[0],
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=1000) for _ in range(2)])
strategy = tf.distribute.MirroredStrategy()

# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

from definitions import DATA_DIR, TRAIN_LOGS_DIR
from src.autoencoders.convolutional_small_autoencoder import ConvolutionalSmallAutoencoder
from src.datasets.genome_window_dataset_multigpu import DistributedGenomeWindowDataset
from src.metrics.locality_preserving_loss_multigpu import LocalityPreservingLossMultigpu

per_worker_batch_size = 4096
global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync

n_epochs = 2

# dataset parameters
data_dir = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced_small")
window_size = 75
step_size = 3
n_mutations = 1

train_dataset = DistributedGenomeWindowDataset(data_dir, "train", window_size, step_size, global_batch_size,
                                               n_mutations, shuffle=False)
val_dataset = DistributedGenomeWindowDataset(data_dir, "val", window_size, step_size, global_batch_size,
                                             n_mutations, shuffle=False)

# autoencoder parameters
latent_dim = 10
autoencoder_name = "multigpu_local_test"

# training parameters
max_seq_weight = 1e-2
max_sim_weight = 1e-4
# max_seq_weight = 0.0
# max_sim_weight = 0.0
n_seq_windows = 3
seq_window_weights = tf.constant([1.0, 0.75, 0.5])
learning_rate = 1e-4
# annealing
n_weight_cycles = 5
weight_cycle_proportion = 0.8

with strategy.scope():
    # datasets
    train_tf_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: train_dataset.instantiate_dataset(global_batch_size, input_context))
    val_tf_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: val_dataset.instantiate_dataset(global_batch_size, input_context))

    # autoencoder and optimizer
    autoencoder = ConvolutionalSmallAutoencoder(window_size, latent_dim, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # metrics
    loss_fn = LocalityPreservingLossMultigpu(n_mutations, train_dataset.global_n_batches, per_worker_batch_size,
                                             max_sim_weight, max_seq_weight, n_seq_windows, seq_window_weights,
                                             n_weight_cycles, weight_cycle_proportion)
    train_loss = tf.keras.metrics.Mean()
    train_rec_loss = tf.keras.metrics.Mean()
    train_seq_loss = tf.keras.metrics.Mean()
    train_sim_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    val_rec_loss = tf.keras.metrics.Mean()
    val_seq_loss = tf.keras.metrics.Mean()
    val_sim_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.CategoricalAccuracy()


@tf.function
def train_step(step, iterator):
    """Training step function."""

    def step_fn(step, inputs):
        """Per-Replica step function."""
        organism_ids, originals = inputs
        with tf.GradientTape() as tape:
            # encode windows and reconstruct input from latent encoding
            encoded = autoencoder.encoder(originals)
            reconstructed = autoencoder.decoder(encoded)

            # compute loss
            per_batch_loss, per_batch_rec_loss, per_batch_seq_loss, per_batch_sim_loss = \
                loss_fn.compute_loss(step, organism_ids, originals, encoded, reconstructed)
            loss = per_batch_loss / strategy.num_replicas_in_sync

        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

        # update metrics
        train_loss.update_state(per_batch_loss)
        train_rec_loss.update_state(per_batch_rec_loss)
        train_seq_loss.update_state(per_batch_seq_loss)
        train_sim_loss.update_state(per_batch_sim_loss)
        train_accuracy.update_state(originals, reconstructed)

        return loss

    per_replica_losses = strategy.run(step_fn, args=(step, next(iterator),))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def val_step(iterator):
    """Validation step function."""

    def step_fn(inputs):
        """Per-replica step function."""
        organism_ids, originals = inputs
        # perform encoding and reconstruction
        encoded = autoencoder.encoder(originals)
        reconstructed = autoencoder.decoder(encoded)

        # calculate loss
        per_batch_loss, per_batch_rec_loss, per_batch_seq_loss, per_batch_sim_loss = \
            loss_fn.compute_loss(tf.constant(0.0, dtype=tf.float32), organism_ids, originals, encoded, reconstructed)

        # update metrics
        val_rec_loss.update_state(per_batch_rec_loss)
        val_seq_loss.update_state(per_batch_seq_loss)
        val_sim_loss.update_state(per_batch_sim_loss)
        val_accuracy.update_state(originals, reconstructed)

        return per_batch_loss

    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


# tensorboard training logs
logs_dir = os.path.join(TRAIN_LOGS_DIR, "autoencoders", autoencoder_name)
train_summary_writer = tf.summary.create_file_writer(logdir=logs_dir)

epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name="epoch")
step_in_epoch = tf.Variable(initial_value=tf.constant(0.0, dtype=tf.dtypes.float32), name="step_in_epoch")
while epoch.numpy() < n_epochs:
    # training loop setup
    pbar = tqdm(total=train_dataset.global_n_batches)
    iterator = iter(train_tf_dataset)
    start = time.time()

    # training loop
    train_dataset.prepare_for_epoch()
    while step_in_epoch.numpy() < train_dataset.global_n_batches:
        train_step(step_in_epoch, iterator)
        step_in_epoch.assign_add(1.0)
        pbar.update(1)

        if step_in_epoch.numpy() % 100 == 0:
            iteration = epoch.numpy() * train_dataset.global_n_batches + step_in_epoch.numpy()
            with train_summary_writer.as_default():
                tf.summary.scalar(f"lp_weight", loss_fn.get_annealed_weight(iteration), step=iteration)
                # tf.summary.scalar(f"train_loss_iters", train_loss.result(), step=iteration)
                # tf.summary.scalar(f"train_reconstruction_loss_iters", train_rec_loss.result(), step=iteration)
                # tf.summary.scalar(f"train_sequentiality_loss_iters", train_seq_loss.result(), step=iteration)
                # tf.summary.scalar(f"train_similarity_loss_iters", train_sim_loss.result(), step=iteration)
                # tf.summary.scalar(f"train_accuracy_iters", train_accuracy.result(), step=iteration)
                # tf.summary.scalar(f"val_reconstruction_loss_iters", val_rec_loss.result(), step=iteration)
                # tf.summary.scalar(f"val_sequentiality_loss_iters", val_seq_loss.result(), step=iteration)
                # tf.summary.scalar(f"val_similarity_loss_iters", val_sim_loss.result(), step=iteration)
                # tf.summary.scalar(f"val_accuracy_iters", val_accuracy.result(), step=iteration)

    # validation loop setup
    pbar.close()
    pbar = tqdm(total=val_dataset.global_n_batches)
    step_in_epoch.assign(0.0)
    iterator = iter(val_tf_dataset)

    # validation loop
    val_dataset.prepare_for_epoch()
    while step_in_epoch.numpy() < val_dataset.global_n_batches:
        val_step(iterator)
        step_in_epoch.assign_add(1.0)
        pbar.update(1)

    # # write metrics
    # total_loss, rec_loss, seq_loss, sim_loss = loss_fn.results()
    # loss_fn.reset()
    # train_acc = train_acc_metric.result()
    # train_acc_metric.reset()
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset()
    # print(f"Epoch {epoch.numpy()} Reconstruction loss {rec_loss:.4e} "
    #       f"Sequentiality loss {seq_loss:.4e} Similarity loss {sim_loss:.4e} "
    #       f"Train accuracy {train_acc * 100:.2f} % Validation accuracy {val_acc * 100:.2f} %")

    # update and reset variables
    epoch.assign_add(1)
    step_in_epoch.assign(0.0)
