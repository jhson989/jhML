import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.layers as nn
import jhML.functions as F
import jhML.optimizer as optim

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


    net = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 1000),
        nn.ReLU(),
        nn.Linear(1000, 800),
        nn.ReLU(),
        nn.Linear(800, 100),
        nn.ReLU(),
        nn.Linear(100, 8),
        nn.ReLU(),
        nn.Linear(8, num_class)
            ).to_gpu()

    #### learning
    optim = optim.RMSprop(net.params(), lr=1e-4)
    num_epoch = int(1e+4)
    for epoch in range(num_epoch):
        data, label  = jhML.Variable(x).to_gpu(), jhML.Variable(gt).to_gpu()
        optim.zero_grad()
        pred = net(data)
        loss = F.softmax_cross_entropy(pred, label)
        loss.backward()
        optim.step()

        if epoch % (num_epoch/100) == 0:
            print("%d/%d" % (epoch, num_epoch))
            print(F.argmax(pred))
            

    for num in x:
        data = jhML.Variable(num).to_gpu()
        pred = net(data).to_cpu()
        print("0b%d%d%d = %d" % (num[0], num[1], num[2], F.argmax(pred, axis=0)))
        print(F.softmax(pred, axis=0).data)



        
