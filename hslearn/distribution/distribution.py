from abc import ABCMeta, abstractmethod
import math
import numpy as np


__all__ = ['GaussianDistribution', 'DiscreteDistribution']


class BaseDistribution(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def param(self):
        pass

    @abstractmethod
    def prob(self, x):
        pass


class DiscreteDistribution(BaseDistribution):

    def __init__(self, vector):
        super().__init__()
        self._vector = vector

    @property
    def param(self):
        return {'count': self.count, 'unique_count': self.unique_count}

    @property
    def count(self):
        return len(self._vector)

    @property
    def unique_count(self):
        return len(set(self._vector))

    @property
    def unique_values(self):
        return set(self._vector)

    def prob(self, x):
        return self.count_x(x)

    def count_x(self, x):
        return len(self._vector[self._vector == x])


class GaussianDistribution(BaseDistribution):

    def __init__(self, vector):
        super().__init__()
        self._mean = np.mean(vector)
        self._variance = np.var(vector)

    @property
    def param(self):
        return {'mean': self._mean, 'var': self._variance}

    def prob(self, x):
        return ((math.e ** (- (x - self._mean) ** 2 / (2 * self._variance))) /
                ((2 * math.pi * self._variance) ** (1/2)))

