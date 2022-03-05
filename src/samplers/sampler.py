from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def sample_np(self, points, point_labels=None):
        pass

    @abstractmethod
    def sample_tf(self, points, point_labels=None):
        pass
