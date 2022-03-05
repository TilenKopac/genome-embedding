import tensorflow as tf


class HybridAutoencoder(tf.keras.Model):

    def __init__(self, latent_dim):
        super(HybridAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_filters_base = 64
        self.kernel_size = 9
        self.strides = 2

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 4)),
                tf.keras.layers.Conv1D(self.n_filters_base, self.kernel_size, strides=2, padding='same',
                                       activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(self.n_filters_base, self.kernel_size, strides=2, padding='same',
                                       activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(self.n_filters_base * 2, self.kernel_size, strides=2, padding='same',
                                       activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(self.n_filters_base * 2, self.kernel_size, strides=2, padding='same',
                                       activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim, activation='relu')
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(8 * self.n_filters_base * 2, activation='relu'),
                tf.keras.layers.Reshape((8, self.n_filters_base * 2)),
                tf.keras.layers.Conv1DTranspose(self.n_filters_base * 2, self.kernel_size, strides=2, padding='same',
                                                activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1DTranspose(self.n_filters_base * 2, self.kernel_size, strides=2, padding='same',
                                                activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1DTranspose(self.n_filters_base, self.kernel_size, strides=2, padding='same',
                                                activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1DTranspose(self.n_filters_base, self.kernel_size, strides=2, padding='same',
                                                activation='elu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))
            ]
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return reconstructed
