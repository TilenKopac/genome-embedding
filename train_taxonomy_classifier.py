import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from definitions import DATA_DIR, TRAIN_LOGS_DIR, CHECKPOINTS_DIR
from src.classifiers.taxonomy_classifier import TaxonomyClassifier
from src.datasets.taxonomy_dataset import TaxonomyDataset, TaxonomicRankEnum
from src.samplers.sampler_enum import SamplerEnum

# dataset parameters
dataset_dir = os.path.join(DATA_DIR, "deepmicrobes_mag_reads")
encoder_name = "661k_conv_small_loc_pres_ld10_ws100"
sampler_name = SamplerEnum.NO_SAMPLER.value
# dataset_dir = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced")
# encoder_name = "661k_conv_small_loc_pres_ld10"
# sampler_name = SamplerEnum.HYPERCUBE_FINGERPRINT_MEDIAN.value

window_size = 100
batch_size = 64
tax_rank = TaxonomicRankEnum.FAMILY

with open(os.path.join(dataset_dir, "taxa_index.pkl"), "rb") as file:
    taxa_index = pickle.load(file)
with open(os.path.join(dataset_dir, "organism_taxa.pkl"), "rb") as file:
    organism_taxa = pickle.load(file)
train_dataset = TaxonomyDataset(dataset_dir, "train", encoder_name, sampler_name, batch_size, taxa_index, organism_taxa,
                                tax_rank, limit=None)

val_dataset = TaxonomyDataset(dataset_dir, "val", encoder_name, sampler_name, batch_size, taxa_index, organism_taxa,
                              tax_rank, limit=None)

# training parameters
learning_rate = 1e-5
n_epochs = 50

classifier_name = "mag-reads-family-classifier"
# classifier_name = "bacteria-family-hypercube-median-normalized-classifier"
classifier = TaxonomyClassifier(train_dataset.n_labels)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# metrics
loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()


@tf.function
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        # predict taxa
        y_pred = classifier(x, training=True)

        # compute loss
        loss_value = loss_fn(y_true, y_pred)

    gradients = tape.gradient(loss_value, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

    # compute accuracy
    train_acc_metric.update_state(y_true, y_pred)

    return loss_value


@tf.function
def val_step(x, y_true):
    y_pred = classifier(x)
    val_acc_metric.update_state(y_true, y_pred)


# tensorboard training logs
train_log_dir = os.path.join(TRAIN_LOGS_DIR, "classifiers", encoder_name, classifier_name)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# model checkpoints
checkpoints_path = os.path.join(CHECKPOINTS_DIR, "classifiers", encoder_name, classifier_name)
checkpoint = tf.train.Checkpoint(classifier=classifier, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoints_path, max_to_keep=None)

# if a checkpoint exists, restore the latest one
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Latest checkpoint restored!")

for epoch in range(n_epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # training loop
    pbar = tqdm(desc=f"Epoch {epoch} training")
    train_dataset.prepare_for_epoch()
    for i, (tax_labels, embeddings) in enumerate(train_dataset.tf_dataset):
        loss_value = train_step(embeddings, tax_labels)
        if i % 100 == 0 and i != 0:
            pbar.update(100)
    pbar.close()

    # validation loop
    pbar = tqdm(desc=f"Epoch {epoch} validation")
    val_dataset.prepare_for_epoch()
    for i, (tax_labels, embeddings) in enumerate(val_dataset.tf_dataset):
        val_step(embeddings, tax_labels)
        if i % 100 == 0 and i != 0:
            pbar.update(100)
    pbar.close()

    # log loss and metrics and reset metrics
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    with train_summary_writer.as_default():
        tf.summary.scalar(f"loss", float(loss_value), step=epoch)
        tf.summary.scalar(f"train_accuracy", float(train_acc), step=epoch)
        tf.summary.scalar(f"val_accuracy", float(val_acc), step=epoch)

    # save model checkpoint
    if (epoch + 1) % 50 == 0:
        checkpoint_path = checkpoint_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {checkpoint_path}")

    print(f"Loss {float(loss_value):.4f} Train accuracy {float(train_acc):.4f} Val accuracy {float(val_acc):.4f}")
    print(f"Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n")
