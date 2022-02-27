from scipy.spatial import ConvexHull

from sampling.random_sampler import RandomSampler
from sampling.sampler import Sampler


class ConvexHullSampler(Sampler):

    def __init__(self, init_sampler=None, name='convex_hull_sampler'):
        self.name = name
        if init_sampler == 'random':
            self.init_sampler = RandomSampler(frac_points=0.1)
        else:
            self.init_sampler = None

    def sample(self, points):
        # sample points using initialization sampler, if defined
        if self.init_sampler and points.shape[0] >= 100:
            points = self.init_sampler.sample(points)

        hull = ConvexHull(points)
        ind = hull.vertices
        return points[ind]
