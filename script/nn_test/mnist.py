import os
import sys
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname(   ( os.path.abspath(__file__)  ) ) ) ))
import jhML
import jhML.layers as nn
import jhML.functions as F
import jhML.optimizer as optim
from jhML.tutorial.dataset import MNIST

def get_accuracy(pred, gt):
    pred = jhML.as_cpu(pred)
    gt = jhML.as_cpu(gt)
    return np.sum((pred == gt).astype(float))/len(gt)
    


net = nn.Sequential(
        nn.Linear(28*28, 28*28),
        nn.ReLU(),
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, MNIST.num_class)
            )

if __name__ == "__main__":

    using_gpu = True
    num_class = 10
    dataset_mnist_train = MNIST(train=True, flatten=True)
    dataset_mnist_test = MNIST(train=False, flatten=True)
    train_dataloader = jhML.Dataloader(dataset_mnist_train, batch_size=256, gpu=using_gpu, drop_last=True)
    test_dataloader = jhML.Dataloader(dataset_mnist_test, batch_size=256, shuffle=False, gpu=using_gpu, drop_last=False)

    net = net.to(using_gpu)

    

    #### learning
    optim = optim.RMSprop(net.params(), lr=5e-5)
    num_epoch = int(1e+2)
    for epoch in range(num_epoch):
        total_loss = 0.0
        total_accuracy = 0.0
        for i, (data, label) in enumerate(train_dataloader):
            optim.zero_grad()
            pred = net(data)
            loss = F.softmax_cross_entropy(pred, label)
            loss.backward()
            optim.step()

            total_loss += loss.data
            total_accuracy += get_accuracy(F.argmax(pred), label.ravel())

            if epoch%1 == 0 and i == (len(train_dataloader)-1):
                print("EPOCH (%d/%d), ITER (%d/%d)" % (epoch, num_epoch, i, len(train_dataloader)))
                print(" acc: %f, loss: %f" % (total_accuracy/len(train_dataloader), total_loss/len(train_dataloader)))
                #pred, label = jhML.as_cpu(pred), jhML.as_cpu(label)
                #print("PR: " + str(F.argmax(pred)))
                #print("GT: " + str(label.ravel()))
            

    print("[[TEST]] Test start")
    total_accuracy = 0.0
    for i, (data, label) in enumerate(test_dataloader):
        pred = net(data)
        total_accuracy += get_accuracy(F.argmax(pred), label.ravel())
    print(" test acc: %f" % (total_accuracy/len(test_dataloader)))

    