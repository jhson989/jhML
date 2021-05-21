import math


class Optimizer:

    def __init__(self, params, lr):
        self.lr = lr
        self.params = params
        self.hooks = []

    def zero_grad(self):
        for param in self.params:
            param.clear_grad()

    def step(self):

        for f in self.hooks:
            f(params)

        for param in self.params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):

    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def update_one(self, param):
        param.data -= self.lr * param.grad

