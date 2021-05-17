import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) )

################################
# [[TEST]] Matmul Linear
################################

if __name__ == "__main__":

    import jhML.functions as F
    from jhML import Variable
    import numpy as np

    num1 = np.random.randn(3, 2)
    num2 = np.random.randn(2, 3)
    num3 = [100, 1000, 10000]

    a = Variable(num1)
    b = Variable(num2)
    c = Variable(num3)

    d = F.linear(a, b, c)
    d.backward()
    print(a)
    print(b)
    print(c)
    print(d)
    print(a.grad)


