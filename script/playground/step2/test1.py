
from core import Variable
from operation import *


a = Variable(2)
b = square(a)
c = square(b)


print(c.data)

