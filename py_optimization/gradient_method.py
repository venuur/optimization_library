'''Implements gradient method from p. 25 on Yurii Nesterov's Introductory Lectures on Convex Optimization.

Requires:
    f: R^n -> R        = Function to be minimized.
    grad_f: R^n -> R^n = Gradient of function to be minimized.
'''

import numpy as np
import numpy.linalg as la

import abc

# %%

class LineSearch(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, x, direction):
        '''Return a new x that is a step in the given direction.'''
        pass


class ConstantLineSearch(LineSearch):
    def __init__(self, stepsize=1.0):
        self.stepsize = stepsize


    def step(self, model, x, direction):
        return x + self.stepsize * direction


class PowerLineSearch(LineSearch):
    def __init__(self, rate=1.0, power=0.5):
        self.rate = rate
        self.power = power
        self.iteration = 0


    def step(self, model, x, direction):
        next_x = x + self.rate / (self.iteration + 1)**self.power * direction
        self.iteration += 1
        return next_x


class GoldsteinArmijoLineSearch(LineSearch):
    # TODO: Research procedures for finding valid steps.
    # For now using version from Wikipedia
    def __init__(self, sufficient_rate, scaling_rate):
        if sufficient_rate >= 1 or scaling_rate >= 1:
            print('Sufficient rate and scaling rate must be strictly less than 1.')
            raise ValueError
        self.sufficient_rate = sufficient_rate
        self.scaling_rate = scaling_rate


    def step(self, model, x, direction):
        stepsize = 1.0
        sufficient_decrease = False

        next_x = x + stepsize * direction
        next_f_val = model.f(next_x)

        while not sufficient_decrease:
            actual_decrease = model.f_val - next_f_val
            estimated_decrease = model.grad_f_val.dot(x - next_x)
            if self.sufficient_rate * estimated_decrease <= actual_decrease:
                sufficient_decrease = True
            else:
                stepsize *= self.scaling_rate * stepsize
                next_x = x + stepsize * direction
                next_f_val = model.f(next_x)

        return next_x

# %%

class GradientMethod():
    def __init__(self,
            linesearch=ConstantLineSearch(0.0001),
            grad_epsilon=1e-6,
            max_iterations=10**6,
            store_trace=False,
            verbose=False):
        self.linesearch = linesearch
        self.grad_epsilon = grad_epsilon
        self.max_iterations = max_iterations
        self.trace = list()
        self.store_trace = store_trace
        self.verbose = verbose

        # Internal State
        self.best_result = None
        self.x = None
        self.f_val = None
        self.grad_f_val = None

        self.f = None
        self.grad_f = None
        self.x_0 = None


    def minimize(self, f, grad_f, x_0):
        '''Minimize function f with gradient grad_f, return result dict.'''

        self.f = f
        self.grad_f = grad_f
        self.x_0 = x_0

        self.x = x_0
        self.f_val = f(self.x)
        self.grad_f_val = grad_f(self.x)

        if self.store_trace:
            self.trace.append(self.dict_from_iteration())

        self.best_result = self.dict_from_iteration()
        self.iteration = 0
        if self.verbose:
            self.print_iteration()

        while la.norm(self.grad_f_val) > self.grad_epsilon and self.iteration < self.max_iterations:
            self.x = self.linesearch.step(self, self.x, -self.grad_f_val)
            self.f_val = f(self.x)
            self.grad_f_val = grad_f(self.x)

            if self.f_val < self.best_result['f']:
                self.best_result = self.dict_from_iteration()

            self.iteration += 1
            if self.verbose:
                self.print_iteration()

            if self.store_trace:
                self.trace.append(self.dict_from_iteration())

        return self.best_result


    def dict_from_iteration(self):
        return {
            'x': self.x,
            'f': self.f_val,
            'grad_f': self.grad_f_val
        }


    def print_iteration(self):
        if self.iteration == 0:
            print('it.', 'x', 'f', 'grad_f', '||grad_f||', sep='\t\t')

        print(self.iteration, self.x, self.f_val, self.grad_f_val, la.norm(self.grad_f_val), sep='\t\t')


def test_f(x):
    '''Rosenbrock function as test.'''
    a = 1
    b = 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


def test_grad_f(x):
    '''Rosenbrock function as test.'''
    a = 1
    b = 100
    grad = [
        -2*(a - x[0]) - 4*b*(x[1] - x[0]**2) * x[0],
        2*b*(x[1] - x[0]**2)
    ]
    return np.array(grad)
