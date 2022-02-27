import numpy as np

from sampling.sampler import Sampler


class RandomSampler(Sampler):

    def __init__(self, num_points=None, frac_points=None, name='random_sampler'):
        if not num_points and not frac_points:
            raise ValueError('You must specify either the num_points or the frac_points argument!')

        self.name = name
        self.num_points = num_points
        self.frac_points = frac_points

    def sample(self, points):
        if self.num_points:
            if points.shape[0] >= self.num_points:
                random_ind = np.random.choice(points.shape[0], size=self.num_points, replace=False)
            else:
                return points
        else:
            num_points = np.floor(self.frac_points * points.shape[0]).astype(np.int32)
            random_ind = np.random.choice(points.shape[0], size=num_points, replace=False)
        return points[random_ind]
