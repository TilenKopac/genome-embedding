import numpy as np
import tensorflow as tf

from src.samplers.convex_hull_sampler import ConvexHullSampler
from src.samplers.sampler import Sampler


class HypercubeFingerprintSampler(Sampler):

    def __init__(self, latent_dim, split_values, init_sampler=None,
                 normalize=False, name='hypercube-signature-sampler'):
        self.name = name
        if init_sampler == 'convex_hull':
            self.init_sampler = ConvexHullSampler()
        else:
            self.init_sampler = None

        self.latent_dim = latent_dim
        self.split_values = split_values
        self.n_cubes = split_values.shape[-1] + 1
        self.normalize = normalize

        self.assign_cube_vec = np.vectorize(self.assign_cube)

    def sample_np(self, points, point_labels=None):
        # sample points using initialization sampler, if defined
        if self.init_sampler:
            points = self.init_sampler.sample_np(points)

        cube_indices = np.apply_along_axis(self.assign_cube, 1, points)

        signature = np.zeros([self.n_cubes] * self.latent_dim, dtype=np.uint32)
        for indices in cube_indices:
            signature[tuple(indices)] += 1

        signature = signature.reshape([1, self.n_cubes ** self.latent_dim])
        if self.normalize:
            signature = signature / np.sum(signature)

        if point_labels is not None:
            label_wrapped = point_labels[0][np.newaxis]
            return label_wrapped, signature
        else:
            return signature

    @tf.function
    def sample_tf(self, points, point_labels=None):
        raise NotImplementedError()

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
