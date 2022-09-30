The CIFAR-10 dataset has been downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html
The model has been quantized to just 8-bit because PyTorch 1.7.0 only supports 8-bit integer quantization, unlike TensorFlow 2.3.0 which supports integer quantization using arbitrary bitwidth from 2 to 16.
