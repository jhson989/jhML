import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.layers as L
import jhML.functions as F
import numpy as np

def clear_grad(net):
    for l in net:
        l.clear_grads()

def update_grad(net, lr=1e-4):
    for l in net:
        for param in l.params():
            if param.grad is not None:
                param.data -= lr*param.grad

def forward(x, net):
    activ = F.relu

    t = jhML.Variable(x, requires_grad=False)
    activated_layers = net[:-1]
    for l in activated_layers:
        t = l(t)
        t = activ(t)
    return net[-1](t)
    
if __name__ == "__main__":

    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    
    gt = [[0],
          [1],
          [1],
          [0]]

    net = [
        L.Linear(2, 4),
        L.Linear(4, 8),
        L.Linear(8, 8),
        L.Linear(8, 1)
    ]

    num_epoch = int(1e+5)
    for epoch in range(num_epoch):
        clear_grad(net)
        pred = forward(x, net)
        loss = F.mean_squared_error(pred, gt)
        loss.backward()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(pred)
        update_grad(net, lr=8e-5)

    for num in x:
        pred = forward([num], net)
        print("%d xor %d = %.4f" % (num[0], num[1], pred.data[0][0]))




        
