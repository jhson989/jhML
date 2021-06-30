from core import Variable
from operation import *

if __name__ == "__main__":

    a = Variable(1.0)
    b = add(a,a)
    c = square(a)
    d = add(a,b)
    e = add(c,d)
    e.backward()
    print(e.data)
    print(a.grad)

    a.clearGrad()

    print(a.grad)

