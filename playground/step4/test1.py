from jhML import Variable


a = Variable(3.0)
b = Variable(2.0)
c = Variable(1.0)

# y = add(mul(a, b), c)
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)

