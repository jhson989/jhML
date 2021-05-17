import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

################################
# [[TEST]] Sum function
################################

if __name__ == "__main__":

    import jhML.functions as F
    from jhML import Variable
    import numpy as np

    num1 = [[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], [[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]]]

    a = Variable(num1)
    b = F.sum(a)
    c = Variable(np.random.randn(*b.shape))
    d = b * c
    d.backward()
    print(a)
    print(b)
    print(c)
    print(d)
    print(a.grad)
    print("\n\n")
    a.clear_grad()

    a = Variable(num1)
    b = F.sum(a, keepdims=True)
    c = Variable(np.random.randn(*b.shape))
    d = b * c
    d.backward()
    print(a)
    print(b)
    print(c)
    print(d)
    print(a.grad)
    print("\n\n")
    a.clear_grad()

    
 
    a = Variable(num1)
    b = F.sum(a, axis=0)
    c = Variable(np.random.randn(*b.shape))
    d = b * c
    d.backward()
    print(a)
    print(b)
    print(c)
    print(d)
    print(a.grad)
    print("\n\n")
    a.clear_grad()

    a = Variable(num1)
    b = F.sum(a, axis=(1,2))
    c = Variable(np.random.randn(*b.shape))
    d = b * c
    d.backward()
    print(a)
    print(b)
    print(c)
    print(d)
    print(a.grad)
    print("\n\n")
    a.clear_grad()




