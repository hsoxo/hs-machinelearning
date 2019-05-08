from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._model = None
        self._dataset = None
        self._feature_names = None
        self._feature_types = None
        self._fitted_model = None

    @property
    def model(self):
        return self._model

    @property
    def x(self):
        return self.features

    @property
    def features(self):
        return self._dataset[:, :-1]

    @property
    def y(self):
        return self.labels

    @property
    def labels(self):
        return self._dataset[:, -1]

    @property
    def feature_count(self):
        return self._dataset.shape[1] - 1

    @property
    def sample_size(self):
        return self._dataset.shape[0]


