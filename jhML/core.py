import weakref
import numpy as np
import jhML

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)



# ===================================
# Configuration
# ===================================

class ProgramConfig:
    """
    Global class variables used for setting Program configurations
    """
    enable_backprop = True
    train = True

def using_config(name, value):
    """
    Try to set [config-value] to [value] and do some prosedures.
    At the end of some code lines, reset [config-value]
    [arg 1] name: attribution name
    [arg 2] value: target value
    """
    old_value = getattr(ProgramConfig, name)
    setattr(ProgramConfig, name, value)
    try:
        yield
    finally:
        setattr(ProgramConfig, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

def test_mode():
    return using_config("train", False)



# ===================================
# Variable
# ===================================

class Variable():
    """
    A data structure for multi-dimensional tensors.
    Auto gradient calculating supported.
    """
    __array_priority__ = 200 # High priority for operators

    def __init__(self, data, requires_grad=True, name=None, array_module=np):

        if data is not None and not isinstance(data, array_types):
            data = array_module.array(data)

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0


    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable (None)"
        return str("Variable ("+str(self.data)+")").replace('\n', '\n' + ' ' * 10)


    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None


    def backward(self, retain_grad=False):

        """
        Calculate gradients for backpropagation automatically
        [arg 1] retain_grad : if True, retain intermediate gradient results. Default value is False
        """


        xp = jhML.compute.get_array_module(self.data)


        if self.grad is None:
            self.grad = xp.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = jhML.compute.as_numpy(self.data)
        return self

    def to_gpu(self):
        if self.data is not None:
            self.data = jhML.compute.as_cupy(self.data)
        return self


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def tensor(obj, array_module=np):
    if isinstance(obj, array_types):
        return obj
    return array_module.array(obj)


### For convenience : an alias (Variable -> Parameter)
class Parameter(Variable):
    pass



# ===================================
# Function
# ===================================

class Function:
    """
    Callable class instance for auto [forward,backward]-propagration
    """
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        if ProgramConfig.enable_backprop: # Remember intermediate outputs
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]


    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()



# ===================================
# Basic scalar (point-wise) arithmetic operations. def-forward and def-backward are only used for calculating.
# forward
#   [args xs] : xp.ndarray type. xs are data
# backward
#   [args gy] : xp.ndarray type. gy is a gradient from a child variable.
# ===================================

class Add(Function):
    def forward(self,x0, x1):
        return x0+x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy, gy
        if x0.shape != x1.shape:  # for broadcast
            gx0 = jhML.functions.sum_to(gx0, x0.shape)
            gx1 = jhML.functions.sum_to(gx1, x1.shape)
        return gx0, gx1
            
class Mul(Function):
    def forward(self,x0, x1):
        return x0 * x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy*x1, gy*x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = jhML.functions.sum_to(gx0, x0.shape)
            gx1 = jhML.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self,gy):
        return -gy

class Sub(Function):
    def forward(self,x0, x1):
        return x0-x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy, -gy
        if x0.shape != x1.shape:  # for broadcast
            gx0 = jhML.functions.sum_to(gx0, x0.shape)
            gx1 = jhML.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Div(Function):
    def forward(self,x0, x1):
        return x0/x1
    def backward(self,gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy/x1, gy * (-x0/x1**2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x0):
        return x0 ** self.c
    def backward(self, gy):
        x0= self.inputs[0].data
        return gy * self.c * (x0**(self.c-1))


def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    return Sub()(x1, x0)

def div(x0, x1):
    return Div()(x0, x1)

def rdiv(x0, x1):
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    ###TODO matmul, dot, max, min
