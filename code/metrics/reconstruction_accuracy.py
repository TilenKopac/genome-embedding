import tensorflow as tf


class ReconstructionAccuracy:

    def __init__(self):
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    def compute_accuracy(self, original, reconstructed):
        self.accuracy(original, reconstructed)

    def result(self):
        return self.accuracy.result()

    def reset(self):
        self.accuracy.reset_states()
