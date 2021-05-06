from core import Variable
from operation import *

if __name__ == "__main__":

    a = Variable(1.0)
    b = add(a,a)
    b.backward()
    print(b.data)
    print(a.grad)

