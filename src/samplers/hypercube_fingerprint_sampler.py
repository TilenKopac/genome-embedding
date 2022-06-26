import numpy as np
import tensorflow as tf

from src.samplers.convex_hull_sampler import ConvexHullSampler
from src.samplers.sampler import Sampler


class HypercubeFingerprintSampler(Sampler):

    def __init__(self, latent_dim, split_values, init_sampler=None,
                 normalize=False, name='hypercube-fingerprint-sampler'):
        self.name = name
        if init_sampler == 'convex_hull':
            self.init_sampler = ConvexHullSampler()
        else:
            self.init_sampler = None

        self.latent_dim = latent_dim
        self.split_values = split_values
        # self.n_cubes = split_values.shape[-1] + 1
        # todo: generalize to multiple split values per axis - see line above
        self.n_cubes = 2
        self.normalize = normalize

        self.assign_cube_vec = np.vectorize(self.assign_cube)

    def sample_np(self, points, point_labels=None):
        # sample points using initialization sampler, if defined
        if self.init_sampler:
            points = self.init_sampler.sample_np(points)

        cube_indices = np.apply_along_axis(self.assign_cube, 1, points)

        fingerprint = np.zeros([self.n_cubes] * self.latent_dim, dtype=np.uint32)
        for indices in cube_indices:
            fingerprint[tuple(indices)] += 1

        fingerprint = fingerprint.reshape([1, self.n_cubes ** self.latent_dim])
        if self.normalize:
            fingerprint /= points.shape[0]

        if point_labels is not None:
            label_wrapped = point_labels[0][np.newaxis]
            return label_wrapped, fingerprint
        else:
            return fingerprint

    def assign_cube(self, point):
        indices = []
        for i in range(point.shape[0]):
            index = 0
            for j in range(self.split_values[i].shape[0]):
                if point[i] < self.split_values[i][j]:
                    break
                index += 1
            indices.append(index)
        return indices

    @tf.function(experimental_relax_shapes=True)
    def sample_tf(self, points, point_labels=None):
        # sample points using initialization sampler, if defined
        if self.init_sampler:
            points = self.init_sampler.sample_tf(points)

        indices = tf.vectorized_map(self.assign_cube_tf, points)

        fingeprint_len = self.n_cubes ** self.latent_dim
        fingerprint = tf.zeros([fingeprint_len], dtype=tf.int32)
        for i in range(0, tf.shape(indices)[-1], 10000):
            fingerprint += tf.cast(tf.reduce_sum(tf.vectorized_map(lambda index: tf.one_hot(index, depth=fingeprint_len,
                                                                                            dtype=tf.int16),
                                                                   indices[i:i + 10000]), axis=0), dtype=tf.int32)

        if self.normalize:
            fingerprint /= tf.shape(points)[0]

        if point_labels is not None:
            label_wrapped = point_labels[0][tf.newaxis]
            return label_wrapped, fingerprint
        else:
            return fingerprint

    @tf.function
    def assign_cube_tf(self, point):
        powers_of_two = tf.ones([self.latent_dim], dtype=tf.int32) * 2 ** tf.range(self.latent_dim)
        # todo: generalize to multiple split values per axis
        indices = tf.cast(point > self.split_values, tf.int32)
        return tf.reduce_sum(indices * powers_of_two)
