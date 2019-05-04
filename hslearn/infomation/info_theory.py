from collections import Counter
import math
import numpy as np

__all__ = ['gini', 'gini_gain',
           'entopy', 'information_gain', 'information_ratio_gain']


def gini(dataset):
    labels = dataset[:, -1]
    counts = np.array(list(Counter(labels).values()))
    return 1 - np.sum(np.square(counts / len(labels)))


def gini_gain(old_gini, dataset, splited_datasets, *args):
    new_gini = 0
    for label, subset in splited_datasets.items():
        new_gini += len(subset) / len(dataset) * gini(subset)
    return old_gini - new_gini


def entopy(dataset):
    """entropy = - \sum_{i=1}^{n} p(x_i) \log_{2} p(x_i)"""
    labels = dataset[:, -1]
    m, _ = dataset.shape
    p = {i: j / m for i, j in Counter(labels).items()}

    def single_ent(a: np.array):
        return - p[a[0]] * math.log(p[a[0]], 2)

    return np.sum(np.apply_along_axis(single_ent, 0, labels))


def information_gain(old_entopy, dataset, splited_datasets, *args):
    new_entopy = sum([entopy(s) * len(s) / len(dataset)
                      for s in splited_datasets.values()])
    return old_entopy - new_entopy


def information_ratio_gain(old_entopy, dataset, splited_datasets, feat_values, *args):
    gain = information_gain(old_entopy, dataset, splited_datasets)
    split_prob = [count / len(dataset) for _, count in Counter(feat_values).items()]
    split_info = sum(map(lambda x: - x * math.log(x, 2), split_prob))
    return gain / split_info

