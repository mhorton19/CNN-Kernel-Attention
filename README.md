# CNN-Kernel-Attention
These are some research ideas I had to leverage global information to dynamically produce/weight CNN kernels.  These ideas are similar to squeeze-and-exitation (https://arxiv.org/pdf/1709.01507.pdf) in that they use global average pooling to incorporate global information into the convolution operation.  However, where squeese and exitation uses the global average pool vector to weight the output channels, I tried to use this vector to weight/generate kernels using three different methods.  In each case, I am using a heavily augmented cifar10 dataset (rotations, shears, and flips) to advantage dynamic viewpoints, since a primary goal of these methods is to learn useful invariances/equivariances. 

kernel_weighted_cnn:

In this method, each conolution layer has a parameter containing a set of kernels as well as a parameter containing a vector corresponding to each kernel.  The global average pooling vector is reduced with a fully connected layer, and then a dot-product is performed between the reduced vector and each kernel's corresponding vector parameter.  This is then passed through a sigmoid and used to weight each convolution kernel.  This is intended to allow the supression of irrelevant filters based on viewpoint, which woud ideally allow the network to learn a rough viewpoint invariance.

fc_generated_kernel_cnn:

In this method, each conolution layer has a parameter containing a vector embedding of each kernel.  Additionally, each convolution layer contains a fully-connected network which takes the global vector and a kernel embedding as input, and outputs a kernel.  Specifically, the global vector and kernel embedding are each transformed to larger vectors, the transformed global vector is passed through a sigmoid and multiplied by the transformed kernel embedding, and then the result is transformed back to a smaller vector. This gating mechanism produced superior results to appending the kernel embedding and global vector then passing through a standard feed-forward network. 

dot_product_generated_kernel_cnn:




