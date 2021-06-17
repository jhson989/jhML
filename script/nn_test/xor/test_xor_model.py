import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.functions as F
import jhML.layers as nn
import jhML.optimizer as optim

if __name__ == "__main__":

    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    
    gt = [[0],
          [1],
          [1],
          [0]]

    data = jhML.Variable(x)      

    net = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
            )

    #### learning
    optim = optim.RMSprop(net.params(), lr=1e-4)
    num_epoch = int(1e+5)
    for epoch in range(num_epoch):
        optim.zero_grad()
        pred = net(data)
        loss = F.mean_squared_error(pred, gt)
        loss.backward()
        optim.step()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(pred)

    for num in x:
        data = jhML.Variable(num)
        pred = net(data)
        print("%d xor %d = %.4f" % (num[0], num[1], pred.data[0]))




        
