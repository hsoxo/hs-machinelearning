from collections import Counter
import numpy as np
import operator

from hslearn.model.base import BaseModel


class KNNModel(BaseModel):

    def __init__(self, k: int = 3):
        super().__init__()
        self._model = 'kNN'
        self._k = k
        self._dataset = None
        self._feature_names = None

    def fit(self, dataset, feat_names=None):
        self._dataset = dataset
        self._feature_names = feat_names

    def predict(self, x):
        x = np.array(x).reshape(1, len(x))
        x = np.tile(x, (self.sample_size, 1))
        distance_matrix = self._dataset[:, :-1] - x
        distance = np.apply_along_axis(np.linalg.norm, 1, distance_matrix)
        distance = sorted(list(zip(distance, self.labels)), key=operator.itemgetter(0))
        nearest_k = [i[1] for i in distance[: self._k]]
        return Counter(nearest_k).most_common()[0][0]
