"""
Original paper: “ImageNet Classification with Deep Convolutional Neural Networks” by
Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton



In the paper we can read:

[i] “The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.”

[ii] “We applied this normalization after applying the ReLU nonlinearity in certain layers.”

[iii] “If we set s < z, we obtain overlapping pooling. This is what we use throughout our network, with s = 2 and z = 3.”

[iv] “The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels (this is the distance between the receptive field centers of neighboring neurons in a kernel map)."

[v] "The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48.

[vi] "The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer.”

[vii] ”The fourth convolutional layer has 384 kernels of size 3 × 3 × 192 , and the fifth convolutional layer has 256 kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each.”

[viii] "We use dropout in the first two fully-connected layers [...]"



Network architecture
The network consists of 5 Convolutional layers and 3 Fully Connected Layers ([ix])

Max Pooling is applied Between the layers:

1conv-2conv ([v])
2conv-3conv ([vi])
5conv-1fc ([ix])

Before Max Pooling a normalization technique is applied. 
At the paper a normalization method named LRN (Local Response Normalization) was used. 
However, since LRN is not part of the standard tensorflow.keras library and it is not 
in the scope of this section to teach how to write custom layers, we will use another method instead. 
We chose to replace LRN with Batch Normalization for this example.
"""