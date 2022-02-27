import numpy as np

from sampling.sampler import Sampler


class CentroidSampler(Sampler):

    def __init__(self, name='centroid_sampler'):
        self.name = name

    def sample(self, points):
        sampled = np.mean(points, axis=0)
        return sampled[np.newaxis, :]
