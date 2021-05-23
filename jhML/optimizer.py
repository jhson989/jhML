from jhML.core import Variable
import math
import numpy as np





##################################################################3
# Optimizer base class #
# : Update weights of a variable
# : Implement "update_one" procedure for each optimization algorithm
##################################################################3

class Optimizer:

    def __init__(self, params, lr):
        '''
        args:
            params: List of variables to be updated
            lr: learning rate
        '''
        self.lr = lr
        self.params = params
        self.hooks = []

    def zero_grad(self):
        for param in self.params:
            param.clear_grad()

    def step(self):

        for f in self.hooks:
            f(self.params)

        for param in self.params:
            self.update_one(param)

    def update_one(self, param: Variable):
        '''
        arg1:
            param: A variable to be updated
        '''
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)



##################################################################3
# Optimization algorithms
# : SGD / MomentumSGD / AdaGrad
##################################################################3

class SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0.0):
        '''
        Stochastic Gradient Descent
        '''
        super().__init__(params, lr)
        self.weight_decay = weight_decay

    def update_one(self, param: Variable):
        '''
        arg1:
            param: A variable to be updated
        '''

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data

        param.data -= self.lr * param.grad


class MomentumSGD(Optimizer):

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        '''
        Stochastic Gradient Descent with Momentum
        '''
        super().__init__(params, lr)
        self.m = momentum
        self.v = {}
        self.weight_decay = weight_decay

    def update_one(self, param: Variable):
        '''
        arg1:
            param: A variable to be updated
        '''

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data
            
        key = id(param)
        if key not in self.v:
            self.v[key] = np.zeros_like(param.data)

        self.v[key] = self.m * self.v[key] - self.lr * param.grad
        param.data += self.v[key]

class AdaGrad(Optimizer):

    def __init__(self, params, lr, weight_decay=0.0, eps=1e-8):
        '''
        Adaptively adjust learning rates for each parameter
        '''
        super().__init__(params, lr)
        self.h = {}
        self.eps = eps
        self.weight_decay = weight_decay

    def update_one(self, param: Variable):
        '''
        arg1:
            param: A variable to be updated
        '''
        key = id(param)
        if key not in self.h:
            self.h[key] = np.zeros_like(param.data)

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data

        self.h[key] = self.h[key] + param.grad * param.grad
        param.data -= self.lr * param.grad / ( self.eps + np.sqrt(self.h[key]) )





##################################################################3
# Hook functions
# ClipGrad, FreezeParam
##################################################################3
