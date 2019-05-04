import numpy as np
from typing import Dict

from hslearn.model.base import BaseModel
from hslearn.distribution import *


DISCRETE = 'd'
CONTINUOUS = 'c'


class FeatureClassifier(object):

    def __init__(self, feature_vector: np.array, label_vector: np.array, smoothing='laplace'):
        assert feature_vector.shape == label_vector.shape, \
            "vector size not the same"

        self._model_type = DISCRETE
        self._smoothing_method = smoothing

        self._label_dist: DiscreteDistribution = DiscreteDistribution(label_vector)
        self._feature_dist: DiscreteDistribution = DiscreteDistribution(feature_vector)
        self._fitted_model = self._discrete_fitting(feature_vector, label_vector)

    @staticmethod
    def _discrete_fitting(feature_vector, label_vector) -> Dict:
        model = dict()
        for value in set(label_vector):
            model[value] = DiscreteDistribution(feature_vector[label_vector == value])
        return model

    def conditional_prob(self, value, target):
        target_model = self._fitted_model[target]
        if self._smoothing_method == 'laplace':
            return ((target_model.count_x(value) + 1) /
                    (self._label_dist.count_x(target) +
                     (self._feature_dist.count +
                      (0 if value in self._feature_dist.unique_values else 1)) * 1))
        else:
            return ((target_model.count_x(value) + 1) /
                    (self._label_dist.count_x(target)))

    def target_prob(self, target):
        if self._smoothing_method == 'laplace':
            return self._label_dist.count_x(target) / (self._label_dist.count + self._label_dist.unique_count * 1)


class FeatureRegressor(object):

    def __init__(self, feature_vector: np.array, label_vector: np.array, distribution='binomial'):
        assert feature_vector.shape == label_vector.shape, \
            "vector size not the same"

        self._model_type = CONTINUOUS
        self._dist = distribution.lower()

        self._label_dist: DiscreteDistribution = DiscreteDistribution(label_vector)
        self._feature_dist: GaussianDistribution = GaussianDistribution(feature_vector)
        if self._dist in ['binomial', 'gaussian']:
            self._fitted_model = self._binomial_fitting(feature_vector, label_vector)

    @staticmethod
    def _binomial_fitting(feature_vector, label_vector) -> Dict:
        model = dict()
        for value in set(label_vector):
            model[value] = GaussianDistribution(feature_vector[label_vector == value])
        return model

    def conditional_prob(self, value, target):
        return self._fitted_model[target].prob(value)

    def target_prob(self, target):
        return self._label_dist.count_x(target) / (self._label_dist.count + self._label_dist.unique_count)


class NaiveBayesianModel(BaseModel):

    def __init__(self):
        super().__init__()
        self._dataset = None
        self._feature_names = None
        self._feature_types = None
        self._fitted_model = None

    def fit(self, dataset, feat_names=None, feat_types=None):
        self._dataset = dataset
        self._feature_names = feat_names
        self._feature_types = feat_types or [self._feat_type_detect(self._dataset[:, i])
                                             for i, n in enumerate(self._feature_names)]
        self._fitted_model = self.build()

    @staticmethod
    def _feat_type_detect(array, alpha=5):
        try:
            array = array.astype(float)
            if np.all(array == array.astype(int)) and len(set(array)) <= alpha:
                return DISCRETE
            else:
                return CONTINUOUS
        except ValueError:
            return DISCRETE

    def build(self):
        model = dict()
        for index, feat_name in enumerate(self._feature_names):
            if self._feature_types[index] == DISCRETE:
                model[feat_name] = FeatureClassifier(self._dataset[:, index], self.labels, smoothing='')
            elif self._feature_types[index] == CONTINUOUS:
                model[feat_name] = FeatureRegressor(self._dataset[:, index].astype(float), self.labels)
        return model

    def predict(self, vector):
        max_prob = 0
        result = None
        for target in set(self.labels):
            prob = len(self.labels[self.labels == target]) / self.sample_size
            for index, value in enumerate(vector):
                feat_name = self._feature_names[index]
                prob *= m._fitted_model[feat_name].conditional_prob(value, target)
            if prob > max_prob:
                max_prob = prob
                result = target
        return result


if __name__ == '__main__':
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    features = iris['data']
    labels = iris['target']
    dataset = np.concatenate([features, labels.reshape(labels.shape[0], 1)], axis=1)
    feature_names = iris['feature_names']

    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(dataset[:, :-1], dataset[:, -1])

    m = NaiveBayesianModel()
    m.fit(dataset, feature_names)

    r = np.random.rand(100, 4)
    r = r * np.array([max(dataset[:, i]) for i in range(len(feature_names))])
    for i in r:
        p1 = model.predict(i.reshape(1,4))
        p2 = m.predict(i)
        print(p1[0], p2)

    # from hslearn.dataset.marry import dataset, feature_names
    # m = NaiveBayesianModel()
    # m.fit(dataset, feature_names)
    # res = m.predict(['帅', '爆好', '高', '上进'])
    # print(res)