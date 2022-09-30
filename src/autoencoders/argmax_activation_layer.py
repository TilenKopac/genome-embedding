import tensorflow as tf


class ArgmaxActivationLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(ArgmaxActivationLayer, self).__init__()

    def call(self, inputs):
        return tf.one_hot(tf.math.argmax(inputs, axis=-1), depth=inputs.shape[-1])
