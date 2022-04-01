import tensorflow as tf


class ConvolutionalSmallAutoencoder(tf.keras.Model):

    def __init__(self, latent_dim):
        super(ConvolutionalSmallAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_filters = 32
        self.kernel_size = 9
        self.strides = 4

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(self.n_filters, self.kernel_size, strides=self.strides, padding="same",
                                       activation="elu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(self.n_filters, self.kernel_size, strides=self.strides, padding="same",
                                       activation="elu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim, activation="relu")
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense((2 * self.strides) * self.n_filters, activation="relu"),
                tf.keras.layers.Reshape(((2 * self.strides), self.n_filters)),
                tf.keras.layers.Conv1DTranspose(self.n_filters, self.kernel_size, strides=self.strides, padding="same",
                                                activation="elu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1DTranspose(self.n_filters, self.kernel_size, strides=self.strides, padding="same",
                                                activation="elu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(4, self.kernel_size, padding="same", activation="softmax")
            ]
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return reconstructed
