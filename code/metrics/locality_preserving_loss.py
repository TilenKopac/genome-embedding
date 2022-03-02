import tensorflow as tf


class LocalityPreservingLoss:

    def __init__(self, n_mutations, n_batches, batch_size, max_sim_weight, max_seq_weight, n_seq_windows, seq_window_weights,
                 n_weight_cycles, weight_cycle_proportion):
        self.n_mutations = n_mutations
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.max_sim_weight = max_sim_weight
        self.max_seq_weight = max_seq_weight
        self.n_seq_windows = n_seq_windows
        self.seq_window_weights = seq_window_weights
        self.annealing_M = n_weight_cycles
        self.annealing_R = weight_cycle_proportion
        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.rec_loss = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.seq_loss = tf.keras.metrics.Mean(name="sequentiality_loss")
        self.sim_loss = tf.keras.metrics.Mean(name="similarity_loss")

    # numerically stable version of tf.norm
    @tf.custom_gradient
    def norm(self, x):
        y = tf.norm(x, axis=-1, keepdims=False)

        def grad(dy):
            return tf.expand_dims(dy, -1) * (x / (tf.expand_dims(y, -1) + 1e-19))

        return y, grad

    def get_annealed_weight(self, iteration):
        tau = iteration % (self.n_batches / self.annealing_M) / (self.n_batches / self.annealing_M)
        if tau > self.annealing_R:
            return 1.0
        else:
            return iteration % (self.n_batches / self.annealing_M) / (self.annealing_R * self.n_batches / self.annealing_M)

    def compute_loss(self, iteration, organism_ids, original, encoded, reconstructed):
        # reconstruction loss
        rec_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(original, reconstructed))
        self.rec_loss(rec_loss)

        encoded_originals = tf.strided_slice(encoded, [0], [self.batch_size], [self.n_mutations + 1])
        originals_organism_ids = tf.strided_slice(organism_ids, [0], [self.batch_size], [self.n_mutations + 1])

        # sequentiality loss
        seq_loss = 0.0
        unique = tf.unique(organism_ids)[0]
        normalization = 0.0
        for i in tf.range(0, tf.shape(unique)[-1]):
            masked = tf.boolean_mask(encoded_originals, mask=(originals_organism_ids == unique[i]))
            if tf.shape(masked)[0] >= 2 * self.n_seq_windows:
                for j in tf.range(0, self.n_seq_windows):
                    seq_loss += self.seq_window_weights[j] * tf.reduce_mean(self.norm(masked[:-(j + 1)] - masked[j + 1:]))
                    normalization += self.seq_window_weights[j]
        seq_loss /= normalization
        self.seq_loss(seq_loss)

        # similarity loss
        sim_loss = 0.0
        normalization = 0.0
        for i in tf.range(1, self.n_mutations + 1):
            encoded_mutations = tf.strided_slice(encoded, [i], [self.batch_size], [self.n_mutations + 1])
            sim_loss += tf.reduce_mean(self.norm(encoded_mutations - encoded_originals))
            normalization += 1.0
        sim_loss /= normalization
        self.sim_loss(sim_loss)

        # seq_weight = sim_weight = get_annealed_weight(iteration) * max_seq_sim_weight
        seq_weight = self.get_annealed_weight(iteration) * self.max_seq_weight
        sim_weight = self.get_annealed_weight(iteration) * self.max_sim_weight

        # total loss
        loss = rec_loss + seq_weight * seq_loss + sim_weight * sim_loss
        self.total_loss(loss)

        return loss

    def results(self):
        return self.total_loss.result(), self.rec_loss.result(), self.seq_loss.result(), self.sim_loss.result()

    def reset(self):
        for metric in [self.total_loss, self.rec_loss, self.seq_loss, self.sim_loss]:
            metric.reset_states()
