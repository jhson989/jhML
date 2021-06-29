## cuDNN Developer Guide 요약 
-----  
cuDNN 7.6.5 를 사용한 jhML을 구현하기 위한 cuDNN 공부  
공식 문서 url : https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-developer-guide/index.html  
cuDNN version : cuDNN 7.6.5

## 1. Overview  
-----  
NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks  
지원되는 primitives implementation list
- Convolution forward and backward, including cross-correlation
- Pooling forward and backward
- Softmax forward and backward
- Neuron activations forward and backward:
- Rectified linear (ReLU)
    - Sigmoid
    - Hyperbolic tangent (TANH)
    - Tensor transformation functions
- LRN, LCN and batch normalization forward and backward

