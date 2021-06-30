import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

from core.variable import Variable

class Function():

    def __init__(self):
        pass

    def __call__(self, input):

        if input is None:
            raise  ValueError("input is not allowed to be None.")
        if not isinstance(input, Variable):
            raise  ValueError("input is not allowed to be Non-varialbe.")
 
        self.input = input
        self.output = self.forward(input.data)

        return self.output

    def forward(self, dataIn):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()



class Square(Function):

    def forward(self, dataIn):
        dataOut = dataIn**2
        return Variable(dataOut, self)

    def backward(self, gy):
        dataIn = self.input.data
        gx = 2*dataIn*gy
        return gx
        

class Exp(Function):

    def forward(self, dataIn):
        dataOut = np.exp(dataIn)
        return Variable(dataOut, self)

    def backward(self, gy):
        dataIn = self.input.data
        gx = np.exp(dataIn)*gy
        return gx

