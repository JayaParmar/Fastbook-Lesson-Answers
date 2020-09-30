# Lesson 13 Convolutions

## What is a "feature"?

A feature is a transformation of data which is designed to make it easier to model.

## Write out the convolutional kernel matrix for a top edge detector.

top\_edge = tensor(\[-1,-1,-1\],

\[0,0,0\],

\[1,1,1\]).float()

## Write out the mathematical operation applied by a 3×3 kernel to a single pixel in an image.

img\_tensor(\[row-1:row+2, col-1:col+2\]) \* top\_edge

## What is the value of a convolutional kernel applied to a 3×3 matrix of zeros?

Zero

## What is "padding"?

Padding is adding additional pixels added around the outside of our image. Most commonly, pixels of zeros are added.

## What is "stride"?

Stride is the jump/move that the kernel should take over the grid.

## Create a nested list comprehension to complete any task that you choose.

tens = tensor(\[for (i,j) in range(1,27) \[for i in range(1,27)\]\])

## What are the shapes of the input and weight parameters to PyTorch's 2D convolution?

Shape of input is (minibatch, in\_channels, iH, iW)

Shape of weight (which is the filter) is (in\_channels, out\_channels, kH, kW)

## What is a "channel"?

A channel is a single basic color in an image—for regular full-color images there are three channels, red, green, and blue.

## What is the relationship between a convolution and a matrix multiplication?

A convolution is a matrix multiplication between the grid and the filter values.

y = ax where x is the filter and a is the grid weight that generates the activation y.

There are two constraints on this matrix multiplication -

1.  > Some elements are always zero.

2.  > Some elements are tied (forced to always have the same value). These are shared weights in the above equation.

## What is a "convolutional neural network"?

When we create a neural network using convolutions instead of linear layers then the architecture is called convolutional neural network.

In Pytorch, this is represented as *nn.Conv2d*

## What is the benefit of refactoring parts of your neural network definition?

First we understand refactoring and then the benefit.

Refactoring is introducing stride in your convolution neural network.

Without stride, the network will produce the output as a map of activations. (Say a grid of 30X30 with a kernel size 3X3 will give 28X28 activation map without padding)

This output is not useful if we do classification since we need single output activation per image. To achieve this, we make the kernel jump over the grid so that the output *activations* keep *halving* at the same time *doubling* the *filter size*. In other words, *refactoring* the neural network.

Benefit - Refactoring parts of your neural networks makes it much less likely you'll get errors due to inconsistencies in your architectures.

Example - After one stride-2 convolution the activation size will be 14×14, after two it will be 7×7, then 4×4, 2×2, and finally size 1. Filter size will be 4, 8, 16, 32. Increasing filter numbers won't decrease the capacity(information/features) of a layer by too much at a time.

## What is Flatten? Where does it need to be included in the MNIST CNN? Why?

Flatten is a function that converts the pooled feature map into a single column.

In programming, it squeezes the architecture on the weight and height dimensions.

It needs to be included in the last layer in MNIST CNN after the activation is size 1X1.

We need it because the output needs to be a 1-dimensional linear vector. The reason being this vector is fed to a fully connected layer to visualize the output as ‘human readable’.

## What does "NCHW" mean?

The input to a Pytorch model layer has to be in NCHW size where

N=batch size

C=number of channels (filters/features)

H=height of input image

W=weight of input image

## Why does the third layer of the MNIST CNN have 7\*7\*(1168-16) multiplications?

MNIST CNN model summary below

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/model_summary.PNG)

Output shape of the second layer is the input shape of the third layer. In our case it is 64X8X7X7.

At the third Conv2d layer, a grid of 7X7=49 locations with a filter size of 8 produces 1168 parameters. The filter (or channel) has a bias of 8X2=16 at this layer. Ignoring the bias from parameters (1168-16), we get the weights of the third layer.

In order to produce activation at the third layer, weight is multiplied by the input grid locations which in our case is a matrix multiplication of (1168-16)\*(7X7).

## What is a "receptive field"?

The receptive field is the area of an image that is involved in the calculation of a layer. In other words, it is the part of the grid which is multiplied with the filter to extract parameters.

## What is the size of the receptive field of an activation after two stride 2 convolutions? Why?

The general formula of size of receptive field of an activation is -

**(n + 2\*pad - ks)//stride + 1**, where pad is the padding, ks, the size of our kernel, and stride is the stride and n is the number of cells on each dimension.

In our example,

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/convolution.PNG)

At convolution 1,

n=22

ks=3

stride=2

pad = 0

Receptive field 1 = (22-3)//2+1 = 10

At convolution 2,

n=10

ks=3

stride=2

pad = 0

Receptive field 2 = (10-3)//2+1 = 4

The receptive field after convolution 2 will be 4X4 as we see in the figure above.

## Run *conv-example.xlsx* yourself and experiment with *trace precedents*.

Excel file can be downloaded from below link and trace precedents can be run on it.

[<span class="underline">https://github.com/JayaParmar/DeepLearning\_posts/blob/master/\_posts/conv-example.xlsx</span>](https://github.com/JayaParmar/DeepLearning_posts/blob/master/_posts/conv-example.xlsx)

## Have a look at Jeremy or Sylvain's list of recent Twitter "like"s, and see if you find any interesting resources or ideas there.

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/tweet.PNG)

## How is a color image represented as a tensor?

Color image is an input matrix in NCHW form. C changes from rank 1 for basic color to rank 3 tensor for full-color image.

A *channel* is a single basic color in an image—for regular full-color images there are three channels, red, green, and blue. PyTorch represents an image as a rank-3 tensor, with dimensions \[channels, rows, columns\].

## How does a convolution work with a color input?

Color input is a matrix with positive values between 0-255. We make a 3X3 filter with values \[-1 -1 -1\], \[0 0 0\], \[1 1 1\]. These two filters are first multiplied and the resulting matrix elements are then added. Matrix multiplication of higher color values with 1 value from the filter gets even higher; with 0 values erased and with -1 values get even smaller.

This way the color input will be reproduced with color values as in the input matrix.

## What method can we use to see that data in DataLoaders?

We use the *first* method to see the data in DataLoaders. First will automatically get the first 64 batches of the data loaders (train or valid to be specified). We save this in a variable and see the batch via shape method.

Example-

dls = mnist.dataloaders(path)

xb,yb = first(dls.valid)

xb.shape

## Why do we double the number of filters after each stride-2 conv?

Stride-2 convolution makes the filter jump 2 pixels on the grid. This decreases the activations in the activation map by 4 (2 on height and 2 on width dimension). In order to preserve the capacity at each layer, we double the number of filters.

## Why do we use a larger kernel in the first conv with MNIST (with simple\_cnn)?

The kernel is of size 4X4 in the first conv with MNIST. This is because we chose to reduce the activation map from 28X28 to 14X14. Hence the kernel size increased from 1X1 to 4X4.

## What information does ActivationStats save for each layer?

ActivationStats stores mean, standard deviation and % activations near zero for each layer.

## How can we access a learner's callback after training?

We write the callback’s name in camel\_case after the learner object. For example,

learn.activation\_stats.plot\_layer\_stats(0)

learn.recorder

learn.trainevalCallback

learn.progressCallback

## What are the three statistics plotted by plot\_layer\_stats? What does the x-axis represent?

Believe x-axis represents the number of activations at that layer number

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/activation_stats.PNG)

## Why are activations near zero problematic?

Activations near zero are problematic, because it means we have computation in the model that's doing nothing at all (since multiplying by zero gives zero). When you have some zeros in one layer, they will generally carry over to the next layer which will then create more zeros.

## What are the upsides and downsides of training with a larger batch size?

Upside - Larger batches have gradients that are more accurate, since they're calculated from more data.

Downside - A larger batch size means fewer batches per epoch, which means less opportunities for your model to update weights.

## Why should we avoid using a high learning rate at the start of training?

We should avoid using a high learning rate at the start of the training because then the model will jump around a lot from batch to batch. This will result in diverging training and the losses might not improve.

## What is 1cycle training?

1cycle training is a technique invented by Leslie Smith where the learning rate grows from the minimum value to the maximum value (*warmup*), and one where it decreases back to the minimum value (*annealing*).

## What are the benefits of training with a high learning rate?

## By training with higher learning rates, we train faster—a phenomenon Smith named *super-convergence*.

## By training with higher learning rates, we overfit less because we skip over the sharp local minima to end up in a smoother (and therefore more generalizable) part of the loss.

## Why do we want to use a low learning rate at the end of training?

Once we have found a nice smooth area for our parameters, we want to find the very best part of that area, which means we have to bring our learning rates down again. This is why 1cycle training has a gradual learning rate cooldown.

## What is "cyclical momentum"?

Cyclical momentum is the momentum of the model which follows the cyclical learning rate but in the opposite direction. The momentum is high in the beginning when the learning rate is low. When we increase the learning rate, the momentum starts coming down and finally goes up again when the learning rate is decreased again.

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/momentum.PNG)

## What callback tracks hyperparameter values during training (along with other information)?

Recorder callback tracks hyperparameter values during training. See for example above, learn.recorder.plot\_sched() tracks learning rate hyperparameter.

## What does one column of pixels in the color\_dim plot represent?

Vertical axis or column represents a group (bin) of activation values. Each column in the horizontal axis is a batch. The colours represent how many activations for that batch have a value in that bin.

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/color_dim.jpeg)

## What does "bad training" look like in color\_dim? Why?

![](https://github.com/JayaParmar/DeepLearning_posts/blob/master/images/bad%20training.png)

We start with nearly all activations at zero—that's what we see at the far left, with all the dark blue. The bright yellow at the bottom represents the near-zero activations. Then, over the first few batches we see the number of nonzero activations exponentially increasing. But it goes too far, and collapses\! We see the dark blue return, and the bottom becomes bright yellow again. It almost looks like training restarts from scratch. Then we see the activations increase again, and collapse again. After repeating this a few times, eventually we see a spread of activations throughout the range.

## What trainable parameters does a batch normalization layer contain?

Batch normalization contains *mean* and *standard deviations* of the activations of a layer. These are used to normalize the activations and get some new activation vector y.

A batchnorm layer returns *gamma*\*y + *beta*. Gamma and beta are two learnable parameters which will be updated in the SGD step.

## What statistics are used to normalize in batch normalization during training? How about during validation?

Layer’s mean and standard deviation along with gamma and beta are the statistics used to normalize in batch normalization during training.

During training, we use the mean and standard deviation of the <span class="underline">batch</span> to normalize the data, while during validation we use a running <span class="underline">mean of the statistics calculated during training</span>.

## Why do models with batch normalization layers generalize better?

Batch normalization adds some extra randomness to the training process. Each mini-batch will have a somewhat different mean and standard deviation than other mini-batches (during training). Therefore, the activations will be normalized by different values each time. In order for the model to make accurate predictions, it will have to learn to become robust to these variations. This makes the model generalize better.

## What features other than edge detectors have been used in computer vision (especially before deep learning became popular)?

[<span class="underline">https://arxiv.org/ftp/arxiv/papers/1910/1910.13796.pdf</span>](https://arxiv.org/ftp/arxiv/papers/1910/1910.13796.pdf)

1.  > Scale Invariant Feature Transform (SIFT)

2.  > Speeded up Robust Features (SURF)

3.  > Features from Accelerated Segment Test (FAST)

4.  > Hough Transforms

5.  > Geometric Hashing

These techniques work on less data compared to deep learning but are not as efficient. SURF and SIFT use Support Vector Machines and K Nearest Neighbours to solve computer vision problems.

More challenging computer vision problems like

a.Robotics

b.Augmented Reality

c.3D modelling

d.Motion estimation

e.Video stabilization

f.Motion capture

g.Video processing and

h.Scene understanding can benefit from ‘traditional’ techniques.

## There are other normalization layers available in PyTorch. Try them out and see what works best. Learn about why other normalization layers have been developed, and how they differ from batch normalization.

Below is the documentation for all normalization layers in Pytorch.

[<span class="underline">https://pytorch.org/docs/stable/nn.html\#normalization-layers</span>](https://pytorch.org/docs/stable/nn.html#normalization-layers)

BatchNorm layers are developed to suit the dimensions of input.

Trying InstanceNorm2d on my data gave an accuracy of 0.102800 against the BatchNorm2d accuracy of 0.984900.

## Try moving the activation function after the batch normalization layer in conv. Does it make a difference? See what you can find out about what order is recommended, and why.

Rewriting the conv function as below

def conv(ni, nf, ks=3, act=True):

layers = \[nn.Conv2d(ni, nf, stride=2, kernel\_size=ks, padding=ks//2)\]

if act: layers.append(nn.BatchNorm2d(nf))

layers.append(nn.ReLU())

return nn.Sequential(\*layers)

Gives an accuracy of 0.891300 as below (versus accuracy of 0.984900 when activation function is before BatchNorm2d)

| epoch | train\_loss | valid\_loss | accuracy  | time |
| ----- | ----------- | ----------- | --------- | ---- |
| 0     | 0.351751    | 0.265655    | 0.8913000 | 0:23 |

The performance of the model is different when BatchNorm is placed before or after activation function. This is because the input activation to BatchNorm will have a different distribution.

Putting it after the activation function gives higher accuracy in our cse.

More information on this link [<span class="underline">https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989/12</span>](https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989/12)

## 

## 

## 

## 

## 

## 

## 

## 

#
