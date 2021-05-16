
import jhML
import jhML.functions as F
from jhML import Variable

x = Variable([1.0, 1.0, 1.0, 2.0])
y = Variable([3.0, 3.0, 3.0, 3.0])
xsin = F.sin(x)
z = xsin * y
print(z)
z.backward()
print(x.grad)


