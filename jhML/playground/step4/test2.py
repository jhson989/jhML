from jhML import Variable
import numpy as np


x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)

y = x + 3.0
print(y)

y = 3.0 * x + 1.0
print(y)

