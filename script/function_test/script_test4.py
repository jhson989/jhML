import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) )

################################
# [[TEST]] Sum_to Broadcast_to function
################################

if __name__ == "__main__":

    import jhML.functions as F
    from jhML import Variable
    import numpy as np

    num1 = [[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], [[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]]]

    a = Variable(num1)
    print(a)

    b= F.sum_to(a, [2,1,1])
    print(b)
    b.backward()
    print(a.grad)

    a = Variable([3.1, 1.02, 4.003])
    b = Variable([4.5])
    c = a*b
    c.backward()
    print(a)
    print(b)
    print(c)
    print(a.grad)
    print(b.grad)


    a = Variable([5, 5, 5])
    b = F.broadcast_to(a, [6, 2, 2, 3])
    b.backward()
    print(b)
    print(a.grad)


    a = Variable([5, 5, 5])
    b = F.broadcast_to(a, [6, 2, 2, 3])
    c = F.sum_to(b, [3])
    c.backward()
    print(b)
    print(c)
    print(a.grad)

    a = Variable(np.random.randn(3))
    print(a)
    b = F.broadcast_to(a, [1, 3])
    print(b)
    b = F.broadcast_to(a, [2, 3])
    print(b)
    b = F.broadcast_to(a, [2, 2, 3])
    print(b)
    a = F.sum_to(b, [2, 3])
    print(a)
    a = F.sum_to(b, [3])
    print(a)

