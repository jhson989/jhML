
from core.variable import Variable
from core.arithmetic.pointwise import *



if __name__ == "__main__":

    x = Variable(0.5)
    y = square(x)
    z = exp(y)
    w = square(z)
    w.backward()
    print(y.data)
    print(x.grad)

