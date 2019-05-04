import numpy as np


__all__ = ['data_spliter']


def binary_split(data: list):
    data = list(data)
    dlist = list(range(len(set(data))))
    results = list()
    halflen = len(dlist) / 2
    for i in range(int(halflen)):
        sublen = i + 1
        if sublen == 1:
            results.append([(n,) for n in dlist])
        else:
            pre = results[-1].copy()
            new = list()
            for p in pre:
                for r in (i for i in dlist if i > max(p)):
                    tmp = p + (r,)
                    if sublen != halflen:
                        new.append(tmp)
                    else:
                        if all(set(_ + tmp) != set(dlist) for _ in new):
                            new.append(tmp)
            results.append(new)
    results = sum(results, [])
    results = [(i, tuple(set(dlist) - set(i))) for i in results]
    results = [(tuple([data[ii] for ii in i]), tuple([data[jj] for jj in j])) for i, j in results]
    return results


def data_spliter(dataset, split_index, min_split_instance=0, method='full'):
    if method.lower() == 'full':
        values = set(dataset[:, split_index])
        res = [{'==' + str(v): dataset[dataset[:, split_index] == v] for v in values}]
    elif method.lower() == 'binary':
        values = dataset[:, split_index]
        if len(set(values)) <= 5 and all(float(i) == int(i) for i in values):
            combination = binary_split(values)
            res = [{str(ds1): dataset[np.isin(dataset[:, split_index], ds1)],
                    str(ds2): dataset[np.isin(dataset[:, split_index], ds2)]}
                   for ds1, ds2 in combination]
        else:
            res = [{'< ' + str(v): dataset[dataset[:, split_index] < v],
                    '>=' + str(v): dataset[dataset[:, split_index] >= v]}
                   for v in values]
    else:
        raise AssertionError("method not supportted")

    res = [s for s in res if all(len(ss) >= min_split_instance for ss in s.values())]
    if not res:
        return [{'all': dataset}]
    return res


