import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

################################

################################

test_mode = False
if len(sys.argv) >= 2 and sys.argv[1].lower() == "true":
    test_mode = True
    
if __name__ == "__main__":
    if test_mode:
        import torch as F
        from torch.autograd import Variable
        from torch import tensor
    else : 
        import jhML.functions as F
        from jhML import Variable
        from jhML import tensor

    num1, num2 = tensor(1.0), tensor(1.0)

    a = Variable(num1, requires_grad=True)
    b = Variable(num2)
        
    c = F.sin(a)
    d = F.cos(b)
    e = c**2 + d**2
    e.backward()
    print(e)
    print(a.grad)





