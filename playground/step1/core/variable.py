
import numpy as np

class Variable():

    def __init__(self, data, creator=None):

        if data is None:
            raise  ValueError("data is not allowed to be None.")
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        self.data = data
        self.grad = np.zeros_like(self.data)
        self.creator = creator

    def backward(self):

        self.grad = np.ones_like(self.data)

        funcs = [self.creator] if self.creator is not None else None
        while funcs:
            f = funcs.pop()
            y, x = f.output, f.input
            x.grad += f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
        

        

