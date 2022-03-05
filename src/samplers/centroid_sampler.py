import numpy as np
import tensorflow as tf

from src.samplers.sampler import Sampler


class CentroidSampler(Sampler):

    def __init__(self, name='centroid-sampler'):
        self.name = name

    def sample_np(self, points, point_labels=None):
        centroid = np.mean(points, axis=0)
        centroid_wrapped = centroid[np.newaxis, :]
        if point_labels is not None:
            return np.array([point_labels[0]]), centroid_wrapped
        else:
            return centroid_wrapped

    @tf.function
    def sample_tf(self, points, point_labels=None):
        centroid = tf.reduce_mean(points, axis=0)
        centroid_wrapped = centroid[tf.newaxis, :]
        if point_labels is not None:
            label_wrapped = point_labels[0][tf.newaxis]
            return label_wrapped, centroid_wrapped
        else:
            return centroid_wrapped
