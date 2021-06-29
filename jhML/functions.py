import numpy as np
import jhML
from jhML.core import Function, Variable, as_variable, as_array
from jhML.compute import get_array_module

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# forward
#   [args xs] xp.ndarray data
# backward
#   [args gy] xp.ndarray data
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.sin(x)
        return y
    def backward(self, gy):
        xp = get_array_module(gy)
        x = self.inputs[0].data
        gx = gy * xp.cos(x) # cos(x)
        return gx

class Cos(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.cos(x)
        return y
    def backward(self, gy):
        xp = get_array_module(gy)
        x = self.inputs[0].data
        gx = gy * -1 * xp.sin(x)
        return gx
        
class Tanh(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        gx = gy * (1 - y * y)
        return gx

class Exp(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.exp(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        gx = gy * y
        return gx

class Log(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.log(x)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy / x
        return gx

def sin(x) -> Variable:
    return Sin()(x)
def cos(x) -> Variable:
    return Cos()(x)
def tanh(x) -> Variable:
    return Tanh()(x)
def exp(x) -> Variable:
    return Exp()(x)
def log(x) -> Variable:
    return Log()(x)

# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        y = x.reshape(self.shape)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy.reshape(x.shape)
        return gx

        
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        if self.axes is None:
            inv_axes = range(x.ndim)[::-1]
        else:
            inv_axes = tuple(xp.argsort([ax for ax in self.axes]))
        gx = gy.transpose(self.axes)
        return gx
        
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
        
    def forward(self, x):
        return x[self.slices]
    def backward(self, gy):
        xp = get_array_module(gy)
        x = self.inputs[0].data
        gx = xp.zeros(x.shape, dtype=gy.dtype)
        xp.add.at(gx, self.slices, gy)
        return gx

def reshape(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x, axes=None) -> Variable:
    return Transpose(axes)(x)

def get_item(x, slices) -> Variable:
    return GetItem(slices)(x)

def flatten(x):
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        xp = get_array_module(gy)
        x = self.inputs[0].data
        gy = self.reshape_sum_backward(gy)
        gx = xp.broadcast_to(gy, x.shape)
        return gx

    def reshape_sum_backward(self, gy):    
        x = self.inputs[0].data
        axis = self.axis
        keepdims = self.keepdims
        ndim = len(x.shape)
        tuple_axis = axis
        if axis is None:
            tuple_axis = None
        elif not isinstance(axis, tuple):
            tuple_axis = (axis, )

        if (ndim == 0 or tuple_axis is None or keepdims):
            shape = gy.shape
        else:
            actual_axis = [a if a >=0 else a+ndim for a in tuple_axis]
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)

        gy = gy.reshape(shape)
        return gy

class SumTo(Function):
    def __init__(self, shape):
        self.y_shape = shape
    def forward(self, x):
        return self.sum_to(x)
    def backward(self, gy):
        x = self.inputs[0].data
        return broadcast_to(gy, x.shape).data
    def sum_to(self, x):   
        ndim = len(self.y_shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(self.y_shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

        
class BroadcastTo(Function):
    def __init__(self, shape):
        self.y_shape = shape
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.broadcast_to(x, self.y_shape)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        return sum_to(gy, x.shape).data

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    def backward(self, gy):
        x, W = self.inputs[0].data, self.inputs[1].data
        gx = gy.dot(W.T)
        gW = (x.T).dot(gy)
        return gx, gW


class Linear(Function):
    def forward(self, x, W, b=None):
        y = x.dot(W)
        if b is not None:
            y = y + b
        return y
    def backward(self, gy):
        x, W, b = self.inputs[0].data, self.inputs[1].data, self.inputs[2].data 
        gb = None if b.data is None else sum_to(gy, b.shape).data
        gx = gy.dot(W.T)
        gW = (x.T).dot(gy)
        return gx, gW, gb



def sum(x, axis=None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)

def sum_to(x, shape) -> Variable:
    if shape == x.shape:
        return as_variable(x)
    return SumTo(shape)(x)

def broadcast_to(x, shape) -> Variable:
    if shape == x.shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
    
def average(x, axis=None, keepdims=False) -> Variable:
    x = as_variable(x)
    y = sum(x, axis, keepdims) 
    return y * (y.data.size / x.data.size)
mean = average # alias

def matmul(x, W) -> Variable:
    return MatMul()(x, W)

def linear(x, W, b=None) -> Variable:
    return Linear()(x, W, b)



# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class Sigmoid(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.tanh(x*0.5)*0.5 + 0.5
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        return gy * y * (1-y)

class ReLU(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        mask = x > 0
        gx = gy * mask
        return gx



class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis
    def forward(self, x):
        xp = get_array_module(x)
        y = x-x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

class LogSoftmax(Function):
    def __init__(self, axis):
        self.axis = axis
    def forward(self, x):
        xp = get_array_module(x)
        m = x.max(axis=self.axis, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        s = y.sum(axis=self.axis, keepdims=True)
        xp.log(s, out=s)
        log_z = m+s
        y = x - log_z
        return y
    def backward(self, gy):
        xp = get_array_module(hy)
        y = self.outputs[0]().data
        gx = gy - xp.exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx

class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope
    def forward(self, x):
        y = x.copy()
        y[y<0] *= self.slope
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        mask = (x>0).astype(gy.dtype)
        mask[mask<0] = self.slope
        gx = gy*mask
        return gx


def sigmoid(x):
    return Sigmoid()(x)

def relu(x):
    return ReLU()(x)

def softmax(x, axis=1):
    return Softmax(axis)(x)

def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)

def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)





# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
        
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = diff.sum() / len(diff)
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        diff = x0 - x1
        gx0 = gy * 2 * diff * (1/len(diff))
        gx1 = -gx0
        return gx0, gx1

class SoftmaxCrossEntropy(Function):
    r"""It is useful when training a classification problem with `C` classes.
    The `input` is expected to contain raw, unnormalized scores for each class.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.
    """
    def __init__(self, weight=None):
        self.weight = weight

    def forward(self, x, gt):
        xp = get_array_module(x)
        N = x.shape[0]
        log_z = self.logsumexp(x)
        log_p = x - log_z
        log_p = log_p[xp.arange(N), gt.ravel()]  

        if self.weight is not None:
            weight = self.weights.astype(x.dtype)[gt.ravel()]
            log_p = weight * log_p

        y = -log_p.sum() / xp.float32(N)
        return y

    def backward(self, gy):
        xp = get_array_module(gy)
        x, t = self.inputs[0].data, self.inputs[1].data
        N, num_class = x.shape

        gy *= 1/N
        y = softmax(x)
        t_onehot = xp.eye(num_class, dtype=t.dtype)[t.ravel()]
        gx = (y.data-t_onehot) * gy
        if self.weight is not None:
            weight = self.weights.astype(x.dtype)[t.ravel()]
            gx = gx * weight

        return gx
        
    def logsumexp(self, x, axis=1):
        xp = get_array_module(x)
        m = x.max(axis=axis, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        s = y.sum(axis=axis, keepdims=True)
        xp.log(s, out=s)
        m += s
        return m

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def softmax_cross_entropy(x, gt):
    return SoftmaxCrossEntropy()(x, gt)





# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================

class Dropout(Function):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
    def forward(self, x):
        xp = get_array_module(x)

        if jhML.ProgramConfig.train:
            self.mask = (xp.random.rand(*x.shape) > self.dropout_ratio).astype(x.dtype)
            #self.scale = xp.array(1.0-self.dropout_ratio).astype(x.dtype)
            y = x * self.mask / (1.0-self.dropout_ratio)
            return y
        else:
            return x
    def backward(self, gy):
        xp = get_array_module(gy)

        if jhML.ProgramConfig.train:
            gx = gy * self.mask / (1.0-self.dropout_ratio)
            return gx
        else:
            return gy
# =============================================================================
# max / min / clip / argmax
# =============================================================================
'''
class Max(Function):
    def forward(self, ):
    def backward(self, ):

class Min(Function):
    def forward(self, ):
    def backward(self, ):

'''


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        mask = (self.x_min<=x) * (x<=self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def argmax(x: Variable, axis=1) -> list[int]:
    xp = get_array_module(x.data)
    return xp.argmax(x.data, axis=axis) 



# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from jhML.functions_conv import *
from jhML.core import add
from jhML.core import sub
from jhML.core import rsub
from jhML.core import mul
from jhML.core import div
from jhML.core import neg
from jhML.core import pow

