import math
import numpy as np


class Optimizer:

    def __init__(self, params, lr):
        self.lr = lr
        self.params = params
        self.hooks = []

    def zero_grad(self):
        for param in self.params():
            param.clear_grad()

    def step(self):

        for f in self.hooks:
            f(params)

        for param in self.params():
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)



##################################################################3
# Optimizer algorithm # SGD / MomentumSGD / AdaGrad
##################################################################3

class SGD(Optimizer):

    def __init__(self, params, lr):
        super().__init__(params, lr)

    def update_one(self, param):
        param.data -= self.lr * param.grad


class MomentumSGD(Optimizer):

    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, lr)
        self.m = momentum
        self.v = {}

    def update_one(self, param):
        key = id(param)
        if key not in self.v:
            self.v[key] = np.zeros_like(param.data)

        self.v[key] = self.m * self.v[key] - self.lr * param.grad
        param.data += self.v[key]

class AdaGrad(Optimizer):

    def __init__(self, params, lr, eps=1e-8):
        super().__init__(params, lr)
        self.h = {}
        self.eps = eps

    def update_one(self, param):
        key = id(param)
        if key not in self.h:
            self.h[key] = np.zeros_like(param.data)

        self.h[key] = self.h[key] + param.grad * param.grad
        param.data -= self.lr * param.grad / ( self.eps + np.sqrt(self.h[key]) )




        
