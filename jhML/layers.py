import numpy as np
import weakref
import jhML
from jhML.core import Parameter
import jhML.functions as F




# =============================================================================
# Layer (base class)
# =============================================================================

class Layer:

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)


    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(x) for x in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        params = []
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                for p in obj.params():
                    params.append(p)
                #yield from obj.params()
            else:
                params.append(obj)
                #yield obj
        return params


    def clear_grads(self):
        for param in self.params():
            param.clear_grad()




# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================

class Linear(Layer):

    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self):
        r''' He weight initialization
        Proposed by He et al. , 2015 <https://arxiv.org/abs/1502.01852>
        '''
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y

ReLU = F.ReLU        

class Sequential(Layer):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    To make it easier to understand, here is a small example::
        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        
        for i, layer in enumerate(layers):
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


