import numpy as np
from sklearn.svm import OneClassSVM

from src.samplers.sampler import Sampler


class SvmSampler(Sampler):

    def __init__(self, nu=0.1, name='svm-sampler'):
        self.name = name
        self.nu = nu

    def sample_np(self, points, point_labels=None):
        ocsvm = OneClassSVM(nu=self.nu)
        ocsvm.fit(points)
        if point_labels is not None:
            return np.repeat(point_labels[0], ocsvm.support_vectors_.shape[0]), ocsvm.support_vectors_
        else:
            return ocsvm.support_vectors_

    def sample_tf(self, points, point_labels=None):
        raise NotImplementedError()
