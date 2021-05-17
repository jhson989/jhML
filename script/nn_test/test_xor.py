import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) )
import jhML
import jhML.functions as F
import numpy as np

def clear_grad(parameter):
    for param in parameter:
        param.clear_grad()

def update_grad(parameter, lr=1e-3):
    for param in parameter:
        param.data -= lr*param.grad

def forward(x, parameters):
    W1, b1, W2, b2, W3, b3, W4, b4 = parameters
    t = x
    t = F.relu((F.linear(t, W1, b1)))
    t = F.relu((F.linear(t, W2, b2)))
    t = F.relu((F.linear(t, W3, b3)))
    return F.linear(t, W4, b4)
    
if __name__ == "__main__":

    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    
    gt = [[0],
          [1],
          [1],
          [0]]

    W1 = jhML.Variable(np.random.randn(2, 4))
    b1 = jhML.Variable(np.random.randn(4))

    W2 = jhML.Variable(np.random.randn(4, 12))
    b2 = jhML.Variable(np.random.randn(12))

    W3 = jhML.Variable(np.random.randn(12, 12))
    b3 = jhML.Variable(np.random.randn(12))

    W4 = jhML.Variable(np.random.randn(12, 1))
    b4 = jhML.Variable(np.random.randn(1))


    parameters = [W1, b1, W2, b2, W3, b3, W4, b4]

    num_epoch = int(1e+4)
    for epoch in range(num_epoch):
        clear_grad(parameters)
        pred = forward(x, parameters)
        loss = F.mean_squared_error(pred, gt)
        loss.backward()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(pred)
        update_grad(parameters, lr=8e-5)

    for num in x:
        pred = forward([num], parameters)
        print("%d xor %d = %.4f" % (num[0], num[1], pred.data[0][0]))




        
