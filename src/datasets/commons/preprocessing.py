import copy

import numpy as np
import tensorflow as tf

nuc_integer_enc = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3,
    "N": 4}
valid_chars = set(nuc_integer_enc.keys())
nucleotides = copy.deepcopy(valid_chars)
nucleotides.remove("N")
# N is encoded as all zeros based on the work of Bartoszewicz (https://doi.org/10.1101/2020.01.29.925354)
nuc_onehot_enc = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "G": [0.0, 1.0, 0.0, 0.0],
    "C": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "N": [0.0, 0.0, 0.0, 0.0]}
nuc_onehot_enc_list = tf.constant(list(nuc_onehot_enc.values()), dtype=tf.float32)


def is_sequence_valid(sequence):
    if set(sequence).difference(valid_chars):
        # record contains invalid nucleotide characters
        return False
    return True


def integer_encode(sequence):
    return [nuc_integer_enc[nuc] for nuc in sequence]


@tf.function(experimental_relax_shapes=True)
def split_into_windows(sequence, window_size, step_size):
    n_windows = (tf.shape(sequence)[0] - window_size) // step_size + 1
    indices = tf.reshape(tf.tile(tf.range(0, window_size), [n_windows]), (n_windows, window_size))
    increment = tf.cast(
        tf.linspace([0] * window_size, [(n_windows - 1) * step_size] * window_size, n_windows),
        tf.int32)
    indices += increment
    windows = tf.gather(sequence, indices)
    return windows


@tf.function
def split_into_windows_with_id(id, sequence, window_size, step_size):
    windows = split_into_windows(sequence, window_size, step_size)
    ids = tf.repeat(id, tf.shape(windows)[0])
    return ids, windows


def split_into_windows_np(sequence, window_size, step_size):
    s0, s1 = sequence.strides
    n_windows = (sequence.shape[0] - window_size) // step_size + 1
    windows = np.lib.stride_tricks.as_strided(sequence,
                                              shape=(n_windows, window_size * 4),
                                              strides=(s0 * step_size, s1),
                                              writeable=False).reshape((n_windows, window_size, 4))
    return windows


def split_into_windows_py(sequence, window_size, step_size):
    windows = []
    for i in range(0, len(sequence) - window_size, step_size):
        windows.append(sequence[i:i + window_size])
    return windows


@tf.function(experimental_relax_shapes=True)
def one_hot_encode(inputs, depth, dtype=tf.float32):
    # tf.one_hot encodes "unknown" inputs to a vector of zeros
    #  this means N will get encoded to [0, 0, 0, 0], which is expected
    encoded = tf.one_hot(inputs, depth=depth, dtype=dtype)
    return encoded


@tf.function(experimental_relax_shapes=True)
def one_hot_encode_sequences(batch, dtype=tf.float32):
    return one_hot_encode(batch, len(nucleotides), dtype=dtype)


def one_hot_encode_np(inputs):
    return np.array(list(nuc_onehot_enc.values()), dtype=np.uint8)[np.array(inputs).reshape(-1)]


@tf.function
def mutate_nucleotide(nucleotide):
    if tf.reduce_all(nucleotide - nuc_onehot_enc_list[0] == 0):
        weights = [0.95, 0.03, 0.01, 0.01]
    elif tf.reduce_all(nucleotide - nuc_onehot_enc_list[1] == 0):
        weights = [0.03, 0.95, 0.01, 0.01]
    elif tf.reduce_all(nucleotide - nuc_onehot_enc_list[2] == 0):
        weights = [0.01, 0.01, 0.95, 0.03]
    elif tf.reduce_all(nucleotide - nuc_onehot_enc_list[3] == 0):
        weights = [0.01, 0.01, 0.03, 0.95]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    ind = tf.random.categorical(tf.math.log([weights]), 1, dtype=tf.int32)[0][0]
    return tf.one_hot(ind, depth=len(nucleotides))


@tf.function
def make_mutations(organism_ids, sequences, n_mutations, batch_size, window_size):
    copies = tf.repeat(sequences, [n_mutations], axis=0)
    n_mut_windows = batch_size // (n_mutations + 1) * n_mutations
    mutated = tf.reshape(
        tf.vectorized_map(mutate_nucleotide, tf.reshape(copies, [n_mut_windows * window_size,
                                                                 len(nucleotides)])),
        [n_mut_windows, window_size, len(nucleotides)])

    # stitch originals and mutations into one tensor
    originals_ind = tf.range(0, batch_size, n_mutations + 1, tf.int32)
    mutations_ind = tf.reshape(
        tf.transpose(tf.stack([tf.range(i, batch_size, n_mutations + 1, tf.int32)
                               for i in range(1, n_mutations + 1)])),
        [n_mut_windows])
    stitched = tf.dynamic_stitch([originals_ind, mutations_ind], [sequences, mutated])
    return tf.repeat(organism_ids, n_mutations + 1), stitched
