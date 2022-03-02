import time

import tensorflow as tf
from tqdm import tqdm

from autoencoders.convolutional_small import ConvolutionalSmallAutoencoder
from dataset import Dataset
from metrics.locality_preserving_loss import LocalityPreservingLoss
from metrics.reconstruction_accuracy import ReconstructionAccuracy

# dataset parameters
data_dir = "../data/viruses/fasta/train"
window_size = 128
step_size = 4
batch_size = 1024
n_mutations = 3
if batch_size % (n_mutations + 1) != 0:
    raise ValueError("Batch size has to be divisible by n_mutations + 1!")
# limit = 1000
limit = None

dataset = Dataset(data_dir, window_size, step_size, batch_size, n_mutations, limit)

# autoencoder parameters
latent_dim = 10

# training parameters
max_sim_weight = 1e-4
max_seq_weight = 1e-2
n_seq_windows = 3
seq_window_weights = tf.constant([1.0, 0.75, 0.5])
learning_rate = 1e-4
n_epochs = 2
# annealing
n_weight_cycles = 5
weight_cycle_proportion = 0.8

autoencoder = ConvolutionalSmallAutoencoder(latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# metrics
loss_metric = LocalityPreservingLoss(n_mutations, dataset.n_batches, batch_size, max_sim_weight, max_seq_weight, n_seq_windows,
                                     seq_window_weights, n_weight_cycles, weight_cycle_proportion)
accuracy_metric = ReconstructionAccuracy()


@tf.function
def train_step(iteration, model, inputs, optimizer):
    organism_ids = inputs[0]
    original = inputs[1]

    with tf.GradientTape() as tape:
        # encode windows and reconstruct input from latent encoding
        encoded = model.encoder(original, training=True)
        reconstructed = model.decoder(encoded, training=True)

        # compute loss
        loss = loss_metric.compute_loss(iteration, organism_ids, original, encoded, reconstructed)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # compute accuracy
    accuracy_metric.compute_accuracy(original, reconstructed)


# tensorboard training logs
logs_dir = "../training-logs/"
train_log_dir = logs_dir + "test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# model checkpoints
checkpoints_dir = "../checkpoints/"
checkpoints_path = checkpoints_dir + "test"
checkpoint = tf.train.Checkpoint(autoencoder=autoencoder, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoints_path, max_to_keep=None)

# if a checkpoint exists, restore the latest one
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Latest checkpoint restored!")

for epoch in range(n_epochs):
    start = time.time()

    loss_metric.reset()
    accuracy_metric.reset()

    step_counter = tf.constant(0.0)

    for i, batch in tqdm(enumerate(dataset.tf_dataset), total=dataset.n_batches, desc=f"Epoch {epoch}"):
        train_step(step_counter, autoencoder, batch, optimizer)
        step_counter += 1.0

        if i % 100 == 0:
            with train_summary_writer.as_default():
                total_loss, rec_loss, seq_loss, sim_loss = loss_metric.results()
                accuracy = accuracy_metric.result()
                tf.summary.scalar(f"total_loss_iters", total_loss, step=epoch * dataset.n_batches + i)
                tf.summary.scalar(f"reconstruction_loss_iters", rec_loss, step=epoch * dataset.n_batches + i)
                tf.summary.scalar(f"sequentiality_loss_iters", seq_loss, step=epoch * dataset.n_batches + i)
                tf.summary.scalar(f"similarity_loss_iters", sim_loss, step=epoch * dataset.n_batches + i)
                tf.summary.scalar(f"accuracy_iters", accuracy, step=epoch * dataset.n_batches + i)

    if (epoch + 1) % 1 == 0:
        checkpoint_path = checkpoint_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {checkpoint_path}")

    total_loss, rec_loss, seq_loss, sim_loss = loss_metric.results()
    accuracy = accuracy_metric.result()
    print(f"Epoch {epoch + 1} Reconstruction loss {rec_loss:.4e} "
          f"Sequentiality loss {seq_loss:.4e} Similarity loss {sim_loss:.4e} "
          f"Accuracy {accuracy * 100:.2f}")

    with train_summary_writer.as_default():
        tf.summary.scalar(f"total_loss_epochs", total_loss, step=epoch)
        tf.summary.scalar(f"reconstruction_loss_epochs", rec_loss, step=epoch)
        tf.summary.scalar(f"sequentiality_loss_epochs", seq_loss, step=epoch)
        tf.summary.scalar(f"similarity_loss_epochs", sim_loss, step=epoch)
        tf.summary.scalar(f"accuracy_epochs", accuracy, step=epoch)

    print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")
