import tensorflow as tf


class TaxonomyClassifier(tf.keras.Model):

    def __init__(self, n_labels, n_layers=2, n_units=20):
        super(TaxonomyClassifier, self).__init__()
        self.n_labels = n_labels
        self.n_layers = n_layers
        self.n_units = n_units

        self.model = tf.keras.Sequential()
        for _ in range(self.n_layers):
            self.model.add(tf.keras.layers.Dense(self.n_units, activation="relu"))
            self.model.add(tf.keras.layers.Dropout(0.6))
        self.model.add(tf.keras.layers.Dense(self.n_labels, activation="softmax"))

    def call(self, inputs, training=False):
        pred = self.model(inputs, training=training)
        return pred
