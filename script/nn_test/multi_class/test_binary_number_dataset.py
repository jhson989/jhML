import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
import jhML
import jhML.layers as nn
import jhML.functions as F
import jhML.optimizer as optim
import numpy as np

class BinaryDataset(jhML.Dataset):
    def __init__(self):
        self.x = np.array([[0, 0, 0], #0
                    [0, 0, 1], #1
                    [0, 1, 0], #2
                    [0, 1, 1], #3
                    [1, 0, 0], #4
                    [1, 0, 1], #5
                    [1, 1, 0], #6
                    [1, 1, 1], #7
                    ])
        self.gt = np.array([[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    ])
        self.len = len(self.x)


    def __getitem__(self, index):
        return self.x[index], self.gt[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":

    using_gpu = True
    num_class = 8
    binary = BinaryDataset()
    train_dataloader = jhML.Dataloader(binary, batch_size=4, gpu=using_gpu, drop_last=False)
    test_dataloader = jhML.Dataloader(binary, batch_size=1, shuffle=False, gpu=using_gpu, drop_last=False)

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
            ).to(using_gpu)

    #### learning
    optim = optim.RMSprop(net.params(), lr=1e-4)
    num_epoch = int(1e+4)
    for epoch in range(num_epoch):
        for i, (data, label) in enumerate(train_dataloader):
            optim.zero_grad()
            pred = net(data)
            loss = F.softmax_cross_entropy(pred, label)
            loss.backward()
            optim.step()

            if epoch % (num_epoch/100) == 0 and i==0:
                pred, label = jhML.as_cpu(pred), jhML.as_cpu(label)
                print("%d/%d" % (epoch, num_epoch))
                print(F.argmax(pred))
                print(label.ravel())
            

    for i, (data, label) in enumerate(test_dataloader):
        print(data)
        pred = net(data).to_cpu()
        print("0b%d%d%d = %d" % (data[0][0], data[0][1], data[0][2], F.argmax(pred, axis=0)[0]))
        print(F.softmax(pred, axis=0).data[0])



        
