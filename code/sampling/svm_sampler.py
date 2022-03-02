from sklearn.svm import OneClassSVM

from sampling.sampler import Sampler


class SvmSampler(Sampler):

    def __init__(self, nu=0.1, name='svm_sampler'):
        self.name = name
        self.nu = nu

    def sample(self, points):
        ocsvm = OneClassSVM(nu=self.nu)
        ocsvm.fit(points)
        return ocsvm.support_vectors_
