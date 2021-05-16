import numpy as np
import jhML
from jhML.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, ):
    def backward(self, ):
        
class Cos(Function):
    def forward(self, ):
    def backward(self, ):
        
class Tanh(Function):
    def forward(self, ):
    def backward(self, ):

class Exp(Function):
    def forward(self, ):
    def backward(self, ):

class Log(Function):
    def forward(self, ):
    def backward(self, ):


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def forward(self, ):
    def backward(self, ):
        
class Transpose(Function):
    def forward(self, ):
    def backward(self, ):
        
class GetItem(Function):
    def forward(self, ):
    def backward(self, ):


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def forward(self, ):
    def backward(self, ):
        
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


# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from jhML.functions_conv import *


