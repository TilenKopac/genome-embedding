import tensorflow as tf
from scipy.spatial import ConvexHull

from src.samplers.random_sampler import RandomSampler
from src.samplers.sampler import Sampler


class ConvexHullSampler(Sampler):

    def __init__(self, init_sampler=None, name='convex-hull-sampler'):
        self.name = name
        if init_sampler == 'random':
            self.init_sampler = RandomSampler(frac_points=0.1)
        else:
            self.init_sampler = None

    def sample_np(self, points, point_labels=None):
        # sample points using initialization sampler, if defined
        if self.init_sampler and points.shape[0] >= 100:
            points = self.init_sampler.sample_np(points)

        hull = ConvexHull(points)
        ind = hull.vertices
        if point_labels is not None:
            return point_labels[ind], points[ind]
        else:
            return points[ind]

    @tf.function
    def sample_tf(self, points, point_labels=None):
        raise NotImplementedError()
