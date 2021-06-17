import os
import sys
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) ))
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

    net.load("weights.npz")
    net = net.to(using_gpu)

    
    print("[[TEST]] Test start")
    total_accuracy = 0.0
    for i, (data, label) in enumerate(test_dataloader):
        pred = net(data)
        total_accuracy += get_accuracy(F.argmax(pred), label.ravel())
    print(" test acc: %f" % (total_accuracy/len(test_dataloader)))

    