import numpy as np
import jhML
from jhML.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# forward
#   [args xs] np.ndarray data
# backward
#   [args gy] np.ndarray data
# =============================================================================
class Sin(Function):
    def forward(self, x) -> np.ndarray:
        y = np.sin(x)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * np.cos(x) # cos(x)
        return gx

class Cos(Function):
    def forward(self, x) -> np.ndarray:
        y = np.cos(x)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * -1 * np.sin(x)
        return gx
        
class Tanh(Function):
    def forward(self, x) -> np.ndarray:
        y = np.tanh(x)
        return y
    def backward(self, gy) -> np.ndarray:
        y = self.outputs[0]().data
        gx = gy * (1 - y * y)
        return gx

class Exp(Function):
    def forward(self, x) -> np.ndarray:
        y = np.exp(x)
        return y
    def backward(self, gy) -> np.ndarray:
        y = self.outputs[0]().data
        gx = gy * y
        return gx

class Log(Function):
    def forward(self, x) -> np.ndarray:
        y = np.log(x)
        return y
    def backward(self, gy) -> np.ndarray:
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
    def forward(self, x) -> np.ndarray:
        y = x.reshape(self.shape)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy.reshape(x.shape)
        return gx

        
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
    def forward(self, x) -> np.ndarray:
        y = x.transpose(self.axes)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        if self.axes is None:
            inv_axes = range(x.ndim)[::-1]
        else:
            inv_axes = tuple(np.argsort([ax for ax in self.axes]))
        gx = gy.transpose(self.axes)
        return gx
        
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
        
    def forward(self, x) -> np.ndarray:
        return x[self.slices]
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        gx = np.zeros(x.shape, dtype=gy.dtype)
        np.add.at(gx, self.slices, gy)
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
    def forward(self, x) -> np.ndarray:
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        gy = self.reshape_sum_backward(gy)
        gx = np.broadcast_to(gy, x.shape)
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
    def forward(self, x) -> np.ndarray:
        return self.sum_to(x)
    def backward(self, gy) -> np.ndarray:
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
    def forward(self, x) -> np.ndarray:
        y = np.broadcast_to(x, self.y_shape)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        return sum_to(gy, x.shape).data

class MatMul(Function):
    def forward(self, x, W) -> np.ndarray:
        y = x.dot(W)
        return y
    def backward(self, gy) -> np.ndarray:
        x, W = self.inputs[0].data, self.inputs[1].data
        gx = gy.dot(W.T)
        gW = (x.T).dot(gy)
        return gx, gW


class Linear(Function):
    def forward(self, x, W, b=None) -> np.ndarray:
        y = x.dot(W)
        if b is not None:
            y = y + b
        return y
    def backward(self, gy) -> np.ndarray:
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
        #y = 1 / (1 + np.exp(-x))
        y = np.tanh(x*0.5)*0.5 + 0.5
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        return gy * y * (1-y)

class ReLU(Function):
    def forward(self, x) -> np.ndarray:
        y = np.maximum(x, 0.0)
        return y
    def backward(self, gy) -> np.ndarray:
        x = self.inputs[0].data
        mask = x > 0
        gx = gy * mask
        return gx



class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis
    def forward(self, x):
        y = x-x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
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
        m = x.max(axis=self.axis, keepdims=True)
        y = x - m
        np.exp(y, out=y)
        s = y.sum(axis=self.axis, keepdims=True)
        np.log(s, out=s)
        log_z = m+s
        y = x - log_z
        return y
    def backward(self, gy):
        y = self.outputs[0]().data
        gx = gy - np.exp(y) * gy.sum(axis=self.axis, keepdims=True)
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


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


'''
class SoftmaxCrossEntropy(Function):
    def forward(self, ):
    def backward(self, ):


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================

# =============================================================================
# max / min / clip
# =============================================================================

class Max(Function):
    def forward(self, ):
    def backward(self, ):

class Min(Function):
    def forward(self, ):
    def backward(self, ):

class Clip(Function):
    def forward(self, ):
    def backward(self, ):

'''

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

