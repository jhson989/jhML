import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

################################

################################

if __name__ == "__main__":

    import jhML.functions as F
    from jhML import Variable

    num1 = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    num2 = (1.0, 2.0, 3.0)

    a = Variable(num1, requires_grad=True)
    b = F.reshape(a, (3,2))
    c = F.reshape(a, (2,3))
    d = F.transpose(c)
    e = b * d
    f = F.get_item(e, [1,2])
    g = F.reshape(f,(2,2,1))
    h = F.flatten(g)
    h.backward()
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)
    print(a.grad)





