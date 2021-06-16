import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.layers as L
import jhML.functions as F
import jhML.optimizer as optim

def forward(x, net):
    activ = F.relu

    t = jhML.Variable(x, requires_grad=False)
    activated_layers = net[:-1]
    for l in activated_layers:
        t = l(t)
        t = activ(t)
    return net[-1](t)
    
def params(net):
    params = []
    for l in net:
        for p in l.params():
            params.append(p)

    return params

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

    #### learning
#    optim = optim.SGD(params(net), lr=1e-3)
#    optim = optim.MomentumSGD(params(net), lr=1e-4)
#    optim = optim.AdaGrad(params(net), lr=1e-3, weight_decay=1e-3)
    optim = optim.RMSprop(params(net), lr=1e-4)


    num_epoch = int(1e+5)
    for epoch in range(num_epoch):
        optim.zero_grad()
        pred = forward(x, net)
        loss = F.mean_squared_error(pred, gt)
        loss.backward()
        optim.step()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(pred)

    for num in x:
        pred = forward([num], net)
        print("%d xor %d = %.4f" % (num[0], num[1], pred.data[0][0]))




        
