
import numpy as np

class Variable():

    def __init__(self, data, creator=None, name=None):
    
        if data is None:
            raise  ValueError("data is not allowed to be None.")
        if not isinstance(data, np.ndarray):
            data = np.array(data)
    
        self.data = data
        self.grad = np.zeros_like(self.data)
        self.creator = creator
        self.generation = creator.generation+1 if creator is not None else 0
        self.name = name


    def backward(self):
    
        gy = np.ones_like(self.data)
        self.grad = self.grad+gy 
        if self.creator is None:
            return
 
        funcs = []
        seen_set = set()   
        def add_func(f, gy):
            if f not in seen_set:
                seen_set.add(f)
                funcs.append((f,gy))
                funcs.sort(key= lambda f: f[0].generation)

        add_func(self.creator, gy)
        while funcs:
            f, gy = funcs.pop()
            xs = f.inputs
            gxs = f.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(xs, gxs):
                x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator, gx) ## error ocuured
            
    def clearGrad(self):
        self.grad = np.zeros_like(self.data)
    
class Function():

    def __init__(self):
        pass

    def __call__(self, *inputs):
 
        for input in inputs:
            if input is None:
                raise  ValueError("input is not allowed to be None.") 
            if not isinstance(input, Variable):
                raise  ValueError("input is not allowed to be Non-varialbe.") 


        self.inputs = inputs
        self.generation = max([input.generation for input in self.inputs])

        xs = [x.data for x in inputs]
        output = self.forward(*xs)


        return output

    def forward(self, data):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):

    def forward(self, data):
        output = data**2
        return Variable(output, self)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx

class Add(Function):

    def forward(self, x0, x1):
        output = x0+x1
        return Variable(output, self)       

    def backward(self, gy):
        return gy, gy
    






