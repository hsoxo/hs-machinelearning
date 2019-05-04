import importlib

def load(dataset_name):
    return importlib.import_module('hslearn.dataset.' + dataset_name)


if __name__ == '__main__':
    m = load('marry')
    print(dir(m))