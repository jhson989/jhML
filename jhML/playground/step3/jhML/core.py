
import numpy as np
import contextlib



class Config:
    enable_backprop = True

#contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)



class Variable:
    __array_priority__ = 200

    def __init__(self, data, creator=None, required_grad=True, name=None):
    
        if data is None:
            raise ValueError("Variable should not be None-data")

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data
        if required_grad and Config.enable_backprop: 
            self.grad = np.zeros_like(data)
            
        self.creator = creator
        self.name = name
        self.required_grad = required_grad
        self.generation = creator.generation+1 if creator is not None else 0 


    def backward(self, retain_grad=False):

        gx = np.ones_like(self.data)

        if self.required_grad and retain_grad:
            self.grad = self.grad + gx

        if self.creator is None:
            return
 
        funcs = []
        seen_set = set()   
        def add_func(f, gy):
            if f not in seen_set:
                seen_set.add(f)
                funcs.append((f, gy))
                funcs.sort(key= lambda f: f[0].generation)

        add_func(self.creator, gx)
        while funcs:
            f, gy = funcs.pop()
            xs = f.inputs
            gxs = f.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(xs, gxs):
                if x.required_grad and retain_grad:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator, gx)
            



    

class Function:



class Add(Function):
class Mul(Function):
class Neg(Function):
class Sub(Function):
class Div(Function):
class Pow(Function):


