import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.layers as nn
import jhML.functions as F
import jhML.optimizer as optim
import numpy as np
 

if __name__ == "__main__":

    x = [[0, 0, 0], #0
         [1, 1, 0], #6
         [1, 0, 0], #4
         [0, 1, 1], #3
         [0, 1, 0], #2
         [1, 1, 1], #7
         ]
    
    gt = [[0],
          [6],
          [4],
          [3],
          [2],
          [7],
          ]

    num_class = 8
    data = jhML.Variable(x)      

    net = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, num_class)
            )

    #### learning
    optim = optim.RMSprop(net.params(), lr=1e-4)
    num_epoch = int(1e+4)
    for epoch in range(num_epoch):
        optim.zero_grad()
        pred = net(data)
        loss = F.softmax_cross_entropy(pred, gt)
        loss.backward()
        optim.step()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(F.argmax(pred))
            

    for num in x:
        data = jhML.Variable(num)
        pred = net(data)
        print("0b%d%d%d = %d" % (num[0], num[1], num[2], F.argmax(pred, axis=0)))
        print(F.softmax(pred, axis=0).data)



        
