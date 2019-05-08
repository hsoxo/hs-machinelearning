from collections import namedtuple
import numpy as np
import time
import sympy as sp

from hslearn.model.base import BaseModel


class FirstOrderPolynomial(object):

    def __init__(self, n_variate=1):
        self._n_variate = n_variate

        self._variables = self.__get_variables(self._n_variate)

        self._expr = self.__generate_polynomial(*self._variables)

        self._expr_lambda = sp.lambdify(sum(self._variables, tuple()),
                                        self._expr, "numpy")

        self._cost = CostFunction(self)
        self._theta = None

    # def __repr__(self):
    #     expr = self._expr.copy()
    #     if self._theta is not None:
    #         expr = expr.subs(dict(zip(self.coefficients, self._theta)))
    #     return str(expr)

    def __str__(self):
        expr = self._expr.copy()
        if self._theta is not None:
            expr = expr.subs(dict(zip(self.coefficients, self._theta)))
        return str(expr)

    @property
    def expr(self):
        return self._expr

    @property
    def variables(self):
        return self._variables[1]

    @property
    def coefficients(self):
        return self._variables[0]

    @property
    def cost(self):
        return self._cost

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        assert len(theta) == len(self.coefficients)
        self._theta = theta

    def subs(self, x, theta=None):
        expr = self._expr.copy()
        if theta is not None:
            expr = expr.subs(dict(zip(self.coefficients, theta)))
        elif self._theta is not None:
            expr = expr.subs(dict(zip(self.coefficients, self._theta)))
        expr = expr.subs(dict(zip(self.variables, x)))
        return expr

    def eval(self, theta=None, x=None):
        return self._expr_lambda(*theta, *x)

    @staticmethod
    def __get_variables(n):
        postfix = list(str(i) for i in range(n + 1))
        x_ = sp.symbols(' '.join('x_' + i for i in postfix))[1:]
        theta_ = sp.symbols(' '.join('theta_' + i for i in postfix))# [1:]
        return (theta_, x_,)

    @staticmethod
    def __generate_polynomial(theta, x):
        x = (1,) + x
        return sp.Matrix([theta]).dot(sp.Matrix([x]).T)


class CostFunction(object):

    def __init__(self, function, method='mse'):
        self._h = function
        self._y = sp.symbols('y')
        self._J = getattr(self, method)()
        self._J_lambda = sp.lambdify(sum([self._h.coefficients, self._h.variables, (self._y,)], tuple()),
                                     self._J, 'numpy')

    def gradient(self, theta: np.array, x: np.array, y: np.array):
        m, n = x.shape
        mat = np.concatenate((np.tile(theta, (m, 1)), x, y.reshape(m, 1)), axis=1)
        return (np.array(self._J_lambda(*mat.T)).T ).sum(axis=0) / (2*m)

    def mse(self):
        summation = []
        for xi in self._h.coefficients:
            summation.append(sp.diff((self._h.expr - self._y) ** 2, xi))
        return summation


SDRES = namedtuple('SDRES', ['theta', 'time', 'steps'])


class SteepestDescent(object):

    def __init__(self, target):
        self._target = target
        self._stats = None

    def execute(self, x, y, alpha, theta=None, error=0.001, max_step=10000, verbose=True):
        def iprint():
            print('Step: {:<5}  theta: {}'.format(step, theta))
        tic = time.time()
        step = 0
        if theta is None:
            theta = np.zeros(x.shape[1] + 1)
        grad = self._target.cost.gradient(theta, x, y)
        while not np.all(np.absolute(grad) <= error) and step < max_step:
            theta = theta - alpha * grad
            grad = self._target.cost.gradient(theta, x, y)
            step +=1
            if not verbose:
                iprint()
        toc = time.time() - tic
        self._target.theta = theta
        self._stats = SDRES(theta, toc, step)

    @property
    def theta(self):
        if self._stats is None:
            print("Please execute first")
            return None
        return self._stats.theta

    @property
    def result(self):
        return self._target

    def stat(self):
        if self._stats is None:
            print("Please execute first")
            return None
        print("theta:                   " + str(self._stats.theta))
        print("result function:         " + str(self._target))
        print("time used:               " + str(self._stats.time))
        print("number of itertations:   " + str(self._stats.steps))


class LinearRegressionModel(BaseModel):

    def __init__(self, method='sd'):
        super().__init__()
        self.solver = getattr(self, '_'+method)

    def fit(self, dataset, alpha, feat_names=None, **kwargs):
        self._dataset = dataset
        self._feature_names = feat_names
        self._fitted_model = self.solver(alpha=alpha, **kwargs)

    def predict(self, x):
        return self._fitted_model.subs(x)

    def _sd(self, **kwargs):
        linear_expr = FirstOrderPolynomial(self.feature_count)
        sd = SteepestDescent(linear_expr)
        sd.execute(self.x, self.y, **kwargs)
        sd.stat()
        return sd.result


if __name__ == '__main__':
    x0 = np.arange(0., 10., 0.2)
    m = len(x0)
    x1 = np.full(m, 1.0)
    y = 4 * x0 + 1

    dataset = np.vstack([x0, y]).T   # 通过矩阵变化得到测试集 [x0 x1]

    p = FirstOrderPolynomial(2)

    m = LinearRegressionModel()
    m.fit(dataset, alpha=0.0005)
    print(m._fitted_model)
    print(m.predict([0,1]))
    from sklearn.linear_model import LinearRegression
    import scipy.linalg.lstsq