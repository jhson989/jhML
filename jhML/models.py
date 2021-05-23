
import numpy as np
from jhML.layers import Layer

Model = Layer


class Sequential(Model):

    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        
        for i, layer in enumerate(layers):
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


