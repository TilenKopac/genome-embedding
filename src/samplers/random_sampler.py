import numpy as np
import tensorflow as tf

from src.samplers.sampler import Sampler


class RandomSampler(Sampler):

    def __init__(self, n_points=None, frac_points=None, name='random-sampler'):
        if not n_points and not frac_points:
            raise ValueError('You must specify either the num_points or the frac_points argument!')

        self.name = name
        self.num_points = n_points
        self.frac_points = frac_points

    def sample_np(self, points, point_labels=None):
        if self.num_points:
            if points.shape[0] > self.num_points:
                random_ind = np.random.choice(points.shape[0], size=self.num_points, replace=False)
            else:
                return points
        else:
            num_points = np.floor(self.frac_points * points.shape[0]).astype(np.int32)
            random_ind = np.random.choice(points.shape[0], size=num_points, replace=False)
        if point_labels is not None:
            return point_labels[random_ind], points[random_ind]
        else:
            return points[random_ind]

    @tf.function
    def sample_tf(self, points, point_labels=None):
        raise NotImplementedError()
