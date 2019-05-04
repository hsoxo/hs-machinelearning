from collections import Counter
import math
import numpy as np
import operator

from hslearn.infomation import *
from hslearn.model.base import BaseModel
from hslearn.utility.spliter import data_spliter


class Tree(object):

    def __init__(self, branches: dict, feat_name):
        self.name = feat_name
        self.branches = branches

    @property
    def is_leaf(self):
        if all(not isinstance(v, Tree) for _, v in self.branches.items()):
            return True
        else:
            return False

    @property
    def desc(self):
        res = ["{} : {}".format(label, len(data)) for label, data in self.branches.items()]
        return ' / '.join(res)

    def predict(self, x, feature_list):
        for key, value in self.branches.items():
            if self.name:
                if eval(str(x[feature_list.index(self.name)]) + key):
                    return value.predict(x, feature_list)
            else:
                return max([(k, len(v)) for k, v in self.branches.items()], key=operator.itemgetter(1))[0]


class DecisionTreeModel(BaseModel):

    def __init__(self, model='cart', min_split_instances: int = 1, min_split_info: float=0.1):
        super().__init__()
        assert model.lower() in ['id3', 'c4.5', 'cart'], "Model not Implemented"
        self._model = model.lower()
        self._min_split_info = min_split_info
        self._min_split_instances = min_split_instances
        self._dataset = None
        self._feature_names = None
        self._tree = None
        self._model_desc = {'id3': {'std': entopy,
                                    'cpr': info_gain,
                                    'spm': 'full', },
                            'c4.5': {'std': entopy,
                                     'cpr': ratio_gain,
                                     'spm': 'full'},
                            'cart': {'std': gini,
                                     'cpr': gini_gain,
                                     'spm': 'binary'}}

    @property
    def info_standard(self):
        return self._model_desc[self._model]['std']

    @property
    def info_compare(self):
        return self._model_desc[self._model]['cpr']

    @property
    def split_method(self):
        return self._model_desc[self._model]['spm']

    def fit(self, dataset, feat_names=None):
        self._dataset = dataset
        self._feature_names = feat_names
        self._tree = self.build_tree(self._dataset)

    def split_dataset(self, dataset, avail_feats: list = None):
        """
        """
        info = self.info_standard(dataset)
        if info <= self._min_split_info:
            return None, None
        best_feature = None
        best_gain = 0
        splited_sets = None
        for feat in avail_feats or self._feature_names:
            feat_index = self._feature_names.index(feat)
            for splited_datasets in data_spliter(dataset,
                                                 feat_index,
                                                 self._min_split_instances,
                                                 method=self.split_method):
                if len(splited_datasets) == 1:
                    continue
                gain = self.info_compare(info, dataset, splited_datasets, dataset[:, feat_index])
                if gain > best_gain:
                    best_feature = feat
                    splited_sets = splited_datasets
                    best_gain = gain
        return best_feature, splited_sets

    def build_tree(self, dataset, available_features=None):
        """
        """
        labels = dataset[:, -1]
        if np.all(labels == labels[0]):
            return Tree({labels[0]: dataset}, None)
        split_feature, splited_sets = self.split_dataset(dataset)
        if split_feature:
            return Tree({v: self.build_tree(subset)
                         for v, subset in splited_sets.items()}, split_feature)
        else:
            return Tree(data_spliter(dataset, -1)[0], None)

    @staticmethod
    def tree_to_string(tree, indent=''):
        if tree.is_leaf:  # leaf node
            return tree.desc
        else:
            decision = tree.name
            branches = list()
            for label, data in tree.branches.items():
                if isinstance(data, Tree):
                    branches.append(indent + '\t' + str(label) + ' -> ' +
                                    DecisionTreeModel.tree_to_string(data, indent=indent + '\t'))
                else:
                    branches.append(indent + "{} : {}".format(label, len(data)))
            return '\n' + indent + '\t' + decision + '\n' + '\n'.join(branches)

    def plot_tree(self):
        """Plots the obtained decision tree. """
        print(self.tree_to_string(self._tree).replace('\n\t', '\n').lstrip())

    def predict(self, x):
        return self._tree.predict(x, self._feature_names)


class Prune(object):

    def __init__(self, method, test_set):
        pass


if __name__ == '__main__':
    import sklearn.datasets

    iris = sklearn.datasets.load_iris()
    features = iris['data']
    labels = iris['target']
    dataset = np.concatenate([features, labels.reshape(labels.shape[0], 1)], axis=1)
    feature_names = iris['feature_names']

    m = DecisionTreeModel('cart', min_split_instances=1, min_split_info=0)
    m.fit(dataset, feature_names)
    m.plot_tree()
    print(m.predict(np.array([0.2, 1.3, 5.3, 4.3])))