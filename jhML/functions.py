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
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

class Exp(Function):
    def forward(self, x) -> np.ndarray:
        y = np.exp(x)
        return y
    def backward(self, gy) -> np.ndarray:
        y = self.outputs[0]()
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
'''
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        
class SumTo(Function):
    def forward(self, ):
    def backward(self, ):
        
class BroadcastTo(Function):
    def forward(self, ):
    def backward(self, ):

class MatMul(Function):
    def forward(self, ):
    def backward(self, ):

class Linear(Function):
    def forward(self, ):
    def backward(self, ):


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class Sigmoid(Function):
    def forward(self, ):
    def backward(self, ):
        
class ReLU(Function):
    def forward(self, ):
    def backward(self, ):
        
class Softmax(Function):
    def forward(self, ):
    def backward(self, ):

class LogSoftmax(Function):
    def forward(self, ):
    def backward(self, ):

class LeakyReLU(Function):
    def forward(self, ):
    def backward(self, ):
# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
        
class MeanSquaredError(Function):
    def forward(self, ):
    def backward(self, ):

class SoftmaxCrossEntropy(Function):
    def forward(self, ):
    def backward(self, ):

class LeakyReLU(Function):
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
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow

