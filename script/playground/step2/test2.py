from core import Variable
from operation import *

a = Variable(2)
b = square(a)
c = square(b)
d = square(c)
e = square(d)
e.backward()
print(a.grad)
