import numpy as np

from sampling.convex_hull_sampler import ConvexHullSampler
from sampling.sampler import Sampler


class HypercubeFingerprintSampler(Sampler):

    def __init__(self, latent_dim, split_values, init_sampler=None,
                 normalize=False, name='hypercube_signature_sampler'):
        self.name = name
        if init_sampler == 'convex_hull':
            self.init_sampler = ConvexHullSampler()
        else:
            self.init_sampler = None

        self.num_cubes = 2
        self.latent_dim = latent_dim
        self.split_values = split_values
        self.normalize = normalize

        self.assign_cube_vec = np.vectorize(self.assign_cube)

    def sample(self, points):
        # sample points using initialization sampler, if defined
        if self.init_sampler:
            points = self.init_sampler.sample(points)

        # cube_indices = self.assign_cube_vec(points)
        cube_indices = np.apply_along_axis(self.assign_cube, 1, points)

        signature = np.zeros([self.num_cubes] * self.latent_dim, dtype=np.uint32)
        for indices in cube_indices:
            signature[tuple(indices)] += 1

        signature = signature.reshape([1, self.num_cubes ** self.latent_dim])
        if self.normalize:
            signature = signature / np.sum(signature)

        return signature

    def assign_cube(self, point):
        indices = []
        for i in range(point.shape[0]):
            if point[i] < self.split_values[i]:
                indices.append(0)
            else:
                indices.append(1)
        return indices
