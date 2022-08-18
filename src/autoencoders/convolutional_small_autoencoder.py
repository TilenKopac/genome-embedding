import tensorflow as tf


class ConvolutionalSmallAutoencoder(tf.keras.Model):

    def __init__(self, input_dim, latent_dim, pool_size, n_filters=32, kernel_size=9):
        super(ConvolutionalSmallAutoencoder, self).__init__()
        if input_dim / pool_size % pool_size != 0:
            raise ValueError("Autoencoder input dimension must be wholly divisible by pool size twice. "
                             f"{input_dim} is not a valid dimension for pool_size {pool_size}.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pool_size = pool_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(self.n_filters, self.kernel_size, padding="same", activation="elu"),
                tf.keras.layers.MaxPool1D(self.pool_size, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(self.n_filters, self.kernel_size, padding="same", activation="elu"),
                tf.keras.layers.MaxPool1D(self.pool_size, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim, activation="relu")
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense((self.input_dim / (self.pool_size ** 2)) * self.n_filters, activation="relu"),
                tf.keras.layers.Reshape(((self.input_dim // (self.pool_size ** 2)), self.n_filters)),
                tf.keras.layers.Conv1DTranspose(self.n_filters, self.kernel_size, padding="same", activation="elu"),
                tf.keras.layers.UpSampling1D(self.pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1DTranspose(self.n_filters, self.kernel_size, padding="same", activation="elu"),
                tf.keras.layers.UpSampling1D(self.pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(4, self.kernel_size, padding="same", activation="softmax")
            ]
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return reconstructed
