# Lesson 4 ‘mnist\_basics.ipynb’

## How is a grayscale image represented on a computer? How about a color image?

Grayscale images are stored as a combination of black and white pixels. White pixels are stored as number 0 and black pixels as number 255. Shades of gray are between the two.

Color images are stored as shades of red, green and blue. Darkest red and green or blue as number 255 and lightest (white) as 0. A number value from each of red, green and blue means a color combination on a computer.

## How are the files and folders in the MNIST\_SAMPLE dataset structured? Why?

The MNIST\_SAMPLE dataset follows a common layout for machine learning datasets: separate folders for the training set and the validation set (and/or test set).

There's a folder of 3s, and a folder of 7s which are the labels in the training set. We train the model on the training set and test the model performance on validation set and/or test set.

## Explain how the "pixel similarity" approach to classifying digits works.

We find the average pixel value for every pixel of the 3s, then do the same for the 7s. This will give us two group averages, defining what we might call the "ideal" 3 and 7. Then, to classify an image as one digit or the other, we see which of these two ideal digits the image is most similar to. This is pixel similarity.

## What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

List comprehension creates a plain list of the inputs, in our case single image tensors.

new\_list = \[o\*2 for o in a\_list if o//2\!=0\]

## What is a "rank-3 tensor"?

Rank-3 tensor is a length of a three dimensional shape tensor. First axis in the shape is number of image samples, second axis is the height of the image and the third axis is the width of the image.

## What is the difference between tensor rank and shape? How do you get the rank from the shape? 

The *length* of a tensor's shape is its rank. We get it by finding the length of tensor’s shape as below

Rank = len(tensor.shape)

## What are RMSE and L1 norm?

RMSE is the *root mean squared error.* We take the mean of the *square* of differences (which makes everything positive) and then take the *square root* (which undoes the squaring) of the numbers.

L1 norm is the mean of the *absolute value* of differences (absolute value is the function that replaces negative values with positive values).

## How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

You write a wrapper for a compiled object written (and optimized) in low level language like C. Then you execute the wrapper on a GPU which can make multiple parallel calculations.

## Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

data = \[\[1,2,3\],\[4,5,6\],\[7,8,9\]\]

tns = tensor(data)

new\_tns = tns\*2

new\_tns\[0, 2\], new\_tns\[1,2\], new\_tns\[2,1:2\]

## What is broadcasting?

While one wants to add, subtract, multiply or divide two tensors of different ranks, broadcasting technique will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank.

Broadcasting is a capability that makes tensor code much easier to write.

## Are metrics generally calculated using the training set, or the validation set? Why?

Metrics are calculated using the validation set. This is so that we don't inadvertently overfit—that is, train a model to work well only on our training data.

## What is SGD?

SGD or Stochastic Gradient Descent is a mechanism for learning by updating weights automatically.

Instead of trying to find the similarity between an image and an "ideal image," we could instead look at each individual pixel and come up with a set of weights for each one, such that the highest weights are associated with those pixels most likely to be black for a particular category (3 or 7).

## Why does SGD use mini-batches?

We could use the whole dataset or a single image while performing SGD. But we take a middle way and use a few data items called mini-batch.

Using the whole dataset means you get a more accurate and stable estimate of your dataset's gradients, but it will take longer time to train. Calculating it for a single data item would not use much information, so it would result in a very imprecise and unstable gradient.

Moreover we train the data on an accelerator like GPU perform well if they have lots of work to do at a time. For these two reasons, we use mini-batches to calculate gradients.

## What are the seven steps in SGD for machine learning?

1.  > *Initialize* the weights.

2.  > For each image, use these weights to *predict* whether it appears to be a 3 or a 7.

3.  > Based on these predictions, calculate how good the model is (its *loss*).

4.  > Calculate the *gradient*, which measures for each weight, how changing that weight would change the loss

5.  > *Step* (that is, change) all the weights based on that calculation.

6.  > Go back to step 2, and *repeat* the process.

7.  > Iterate until you decide to *stop* the training process (for instance, because the model is good enough or you don't want to wait any longer).

## How do we initialize the weights in a model?

We initialize the parameters to random values.

## What is "loss"?

Loss is a function that returns a number that is small if the performance of the model is good and large if it is bad.

## Why can't we always use a high learning rate?

We change our parameters (weights on axis x) based on the values of gradients (loss on axis y). This gradient is changed by multiplying it with a small number called learning rate. If the learning rate is too high, the gradient and the loss get higher. The loss ‘bounces’ instead of diverging.

## What is a "gradient"?

Gradient is a differential function which is defined as a small change in the loss function with respect to the small change in weight.

## Do you need to know how to calculate gradients yourself?

No, Pytorch has a function ‘requires\_grad\_()’ which calculaties gradients with respect to a variable at that value.

## Why can't we use accuracy as a loss function?

The gradient of a function is its slope. Accuracy only changes at all when a prediction changes from a 3 to a 7. Accuracy is a function that is constant almost everywhere (except at the threshold, 0.5), so its derivative is nil almost everywhere (and infinity at the threshold). This then gives gradients that are 0 or infinite, which are useless for updating the model.

## Draw the sigmoid function. What is special about its shape?

![](/images/sigmoid.png)

The sigmoid function always outputs a number between 0 and 1.

## What is the difference between a loss function and a metric?

## A metric is to drive human understanding and the loss is to drive automated learning.

## Loss function is a reasonably smooth derivative while metric are the numbers we care about. 

## What is the function to calculate new weights using a learning rate?

## W - = gradient(w) \* lr

## What does the DataLoader class do?

A DataLoader can take any Python collection and convert it into an iterator over many batches

coll = range(15)

dl = DataLoader(coll, batch\_size=5, shuffle=**True**)

list(dl)

## Write pseudocode showing the basic steps taken in each epoch for SGD.

def train\_epoch(model, lr, params):

for xb, yb in dl:

calc\_grad(xb, yb, model)

for p in params:

> p.data - = p.grad.lr

p.grad.zero\_()

## Create a function that, if passed two arguments \[1,2,3,4\] and 'abcd', returns \[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')\]. What is special about that output data structure?

x=\[1,2,3,4\]

y=\['a','b','c','d'\]

for i in zip(x,y):

print(i)

## What does view do in PyTorch?

View is a PyTorch method that changes the shape of a tensor without changing its contents.

## What are the "bias" parameters in a neural network? Why do we need them?

## What does the @ operator do in Python?

@ does matrix multiplication (and not element wise multiplication which is done by \*)

## What does the backward method do?

The "backward" method refers to *backpropagation*, which is the name given to the process of calculating the derivative of each layer.

## Why do we have to zero the gradients?

We make gradients zero at the beginning of every batch because the model learns from the feedback loop by activating loss.backward() i.e. back propagate the gradient so that the model input. Otherwise, things will get very confusing when we try to compute the derivative at the next batch.

## What information do we have to pass to Learner?

We need to pass

  - > the DataLoaders,

  - > the model,

  - > the optimization function (which will be passed the parameters),

  - > the loss function, and

  - > optionally any metrics to print:

## Show Python or pseudocode for the basic steps of a training loop.

**def** train\_epoch(model, lr, params):

**for** xb,yb **in** dl:

calc\_grad(xb, yb, model)

**for** p **in** params:

p.data -= p.grad\*lr

p.grad.zero\_()

## What is "ReLU"? Draw a plot of it for values from -2 to +2.

ReLU is a Rectified Linear Unit. In other words, a function that replaces every negative number with a zero.

![](/images/relu.png)

## What is an "activation function"?

It is the fundamental equation in a neural network

## batch@weights + bias

This equation can be linear or nonlinear.

## What's the difference between F.relu and nn.ReLU? 

nn.ReLU is a PyTorch module that does exactly the same thing as the F.relu function. When using nn.Sequential, PyTorch requires us to use the module version.

## The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

The reason is performance. With a deeper model (that is, one with more layers) we do not need to use as many parameters; it turns out that we can use smaller matrices with more layers, and get better results than we would get with larger matrices, and few layers.

## Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete\! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.

See the solution on https://github.com/JayaParmar/Deep-Learning/blob/master/mnist%200-9%20digits.ipynb
Learner needs to be fixed for the error.
