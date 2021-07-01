from jhML.core import Variable, Parameter
import math
import numpy as np
from jhML.compute import get_array_module




##################################################################3
# Optimizer base class #
# : Update weights of a variable
# : Implement "update_one" procedure for each optimization algorithm
##################################################################3

class Optimizer:

    def __init__(self, params, lr, hooks=[]):
        '''
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            hooks: list of hook function applied to parameters
        '''
        self.lr = lr
        self.params = params
        self.hooks = hooks

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

    def __init__(self, params, lr, weight_decay=0.0, hooks=[]):
        '''
        Stochastic Gradient Descent
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            weight_decay (float): weight decay (L2 penalty) (default: 0)
            hooks: list of hook function applied to parameters
        '''
        super().__init__(params, lr, hooks)
        self.weight_decay = weight_decay

    def update_one(self, param: Parameter):

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data

        param.data -= self.lr * param.grad


class MomentumSGD(Optimizer):

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, hooks=[]):
        '''
        Stochastic Gradient Descent with Momentum
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            momentum (float): momentum factor (default: 0.9)
            weight_decay (float): weight decay (L2 penalty) (default: 0)
            hooks: list of hook function applied to parameters
        '''
        super().__init__(params, lr, hooks)
        self.m = momentum
        self.v = {}
        self.weight_decay = weight_decay

    def update_one(self, param: Parameter):
        xp = get_array_module(param.data)
        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data
            
        key = id(param)
        if key not in self.v:
            self.v[key] = xp.zeros_like(param.data)

        self.v[key] = self.m * self.v[key] - self.lr * param.grad
        param.data += self.v[key]


class AdaGrad(Optimizer):

    def __init__(self, params, lr, weight_decay=0.0, eps=1e-8, hooks=[]):
        '''
        Adaptively adjust learning rates for each parameter
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            eps (float): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float): weight decay (L2 penalty) (default: 0)
            hooks: list of hook function applied to parameters
        '''
        super().__init__(params, lr, hooks)
        self.h = {}
        self.eps = eps
        self.weight_decay = weight_decay

    def update_one(self, param: Parameter):
        xp = get_array_module(param.data)
        key = id(param)
        if key not in self.h:
            self.h[key] = xp.zeros_like(param.data)

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data

        self.h[key] = self.h[key] + param.grad * param.grad
        param.data -= self.lr * param.grad / ( self.eps + xp.sqrt(self.h[key]) )


class RMSprop(Optimizer):

    def __init__(self, params, lr, alpha=0.99, weight_decay=0.0, eps=1e-8, hooks=[]):
        r"""Implements RMSprop algorithm with moving average aproach
        Proposed by G. Hinton <https://arxiv.org/pdf/1308.0850v5.pdf>
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate
            alpha (float): smoothing constant (default: 0.99)
            eps (float): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float): weight decay (L2 penalty) (default: 0)
            hooks: list of hook function applied to parameters
        """
        super().__init__(params, lr, hooks)
        self.v = {}
        self.eps = eps
        self.weight_decay = weight_decay
        self.alpha = alpha

    def update_one(self, param: Parameter):
        xp = get_array_module(param.data)
        key = id(param)
        if key not in self.v:
            self.v[key] = xp.zeros_like(param.data)

        if self.weight_decay != 0.0:
            param.grad += self.weight_decay * param.data

        self.v[key] = self.alpha * self.v[key] + (1-self.alpha) * param.grad * param.grad
        param.data -= self.lr * param.grad / ( self.eps + xp.sqrt(self.v[key]) )




##################################################################3
# Hook functions
# ClipGrad, FreezeParam
##################################################################3

class ClipGrad:
    def __init__(self, max_norm: float, norm_type: int=2):
        r"""Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (int): type of the used p-norm.
        """
        self.max_norm = max_norm        
        self.norm_type = norm_type
        
    def __call__(self, parameters):
        total_norm = 0.0
        for p in parameters:
            total_norm += (p.grad**self.norm_type).sum()
        total_norm = math.sqrt(float(total_norm))

        if total_norm >= self.max_norm:
            rate = self.max_norm/(total_norm + 1e-6)
            for p in parameters:
                p.grad *= rate