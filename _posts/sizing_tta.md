# What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?

ImageNet is a computer vision model containing 1.3 million images of various sizes around 500 pixels across, in 1,000 categories.

Imagenette is a subset of ImageNet containing a subset of 10 very different categories from the original ImageNet dataset, making for quicker training when we want to experiment.

Imagenette is better when one wants to train a model from the scratch and ImageNet is better when one wants to do transfer learning i.e. use the pretrained model and only change the last few layers to train one's own dataset.

# What is normalization?

When training a model, it helps if your input data has a mean of 0 and a standard deviation of 1. This is normalization. In the fastai library one can use Normalize transform inside the DataBlock to do this.

# Why didn't we have to care about normalization when using a pretrained model?

Pretrained model is a model that someone else has trained. If you need to use it for inference or transfer learning, you will need to use the same statistics. Therefore you need to distribute the statistics used for normalization so that others using it can match them.

# What is progressive resizing?

Progressive resizing is an approach there you start training using small images, and end training using large images. Spending most of the epochs training with small images, helps training complete much faster. Completing training using large images makes the final accuracy much higher.

# Implement progressive resizing in your own project. Did it help?

[<span class="underline">https://github.com/JayaParmar/Deep-Learning/blob/master/practice\_sizing\_and\_tta.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/practice_regression.ipynb)

In my project practice sizing and tta, I started training with a batch size of 64 and got an accuracy of

| train\_loss | valid\_loss | accuracy |
| ----------- | ----------- | -------- |
| 1.446041    | 1.199214    | 0.624720 |

Later the training set batch size was increased to 128 and the result was

| train\_loss | valid\_loss | accuracy |
| ----------- | ----------- | -------- |
| 1.112897    | 0.872555    | 0.722181 |

Accuracy improved and the training and validation loss reduced.

# What is test time augmentation? How do you use it in fastai?

During inference or validation, creating multiple versions of each image, using data augmentation, and then taking the average or maximum of the predictions for each augmented version of the image is test time augmentation.

You can pass any DataLoader to fastai's tta method; by default, it will use your validation set:

In my project, training on a dataset of batch size 64 and and total size 224 for two epochs gave an accuracy of

| train\_loss | valid\_loss | accuracy |
| ----------- | ----------- | -------- |
| 0.818131    | 0.720649    | 0.77147  |
| 0.70421     | 0.626576    | 0.79499  |

Later, performing test time augmentation on one item of validation set improved the accuracy to 0.802091121673584.

# Is using TTA at inference slower or faster than regular inference? Why?

Using TTA at inference(validation) is slower than regular inference because it does multiple crops of the single image and passes it through the model to get the maximum/average of predictions for each augmented image.

# What is Mixup? How do you use it in fastai?

Mixup is a very powerful data augmentation technique that can provide dramatically higher accuracy, especially when you don't have much data and don't have a pretrained model that was trained on data similar to your dataset.

We use it in fastai by adding a callback to our Learner.

learn = Learner(dls, model, loss\_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=mixup)

# Why does Mixup prevent the model from being too confident?

Mixup prevents the model from being too confident because we're not showing the same image in each epoch, but are instead showing a random combination of two images.

The labels will be a linear combination of two images resulting in a prediction value between 0 and 1.

# Why does training with Mixup for five epochs end up worse than training without Mixup?

Mixup mixes up two images to modify to a new image which is fed to the model. Everytime the model sees an image which is not similar to the one seen earlier. Therefore, it is harder for the model to understand the training images initially. If we train for five epochs, the model has not learnt to identify the input images. But if we train for more than 80 epochs at least, the model will start to understand the input images and predict better than the models which have not been trained on mixed up images.

# What is the idea behind label smoothing?

The labels of the images are made a bit more than 0 and bit less than 1 instead of being perfect 0 and 1. This makes the model less confident and makes your training more robust, even if there is mislabeled data. The result will be a model that generalizes better.

# What problems in your data can label smoothing help with?

1.   The model learns to assign full probability to the ground-truth label for each training example, it is not guaranteed to generalize.

2.   It encourages the differences between the largest logit and all others to become large, reducing the ability of the model to adapt.

Label smoothing can help counter these two problems and generalize the model better.

# When using label smoothing with five categories, what is the target associated with the index 1?

Here N = 5, number of classes

ϵ = 0.1, parameter which means we are 10% unsure of our labels.

The target associated with the index 1 is 0.96.

In one hot encoded output it is \[0.01, 0.01, 0.01, 0.96, 0.01\]

# What is the first step to take when you want to prototype quick experiments on a new dataset?

You should aim to have an iteration speed of no more than a couple of minutes.

If it's taking longer to do an experiment, think about how you could

  -  cut down your dataset, or

  -  simplify your model,

to improve your experimentation speed.

# Use the fastai documentation to build a function that crops an image to a square in each of the four corners, then implement a TTA method that averages the predictions on a center crop and those four crops. Did it help? Is it better than the TTA method of fastai?

I could not find a method in fastai to do these four crops and center crop but found the ‘FiveCrop’ method in [<span class="underline">pytorch documentation</span>](https://pytorch.org/docs/stable/torchvision/transforms.html). I tried to implement this in my file [<span class="underline">crop & tta.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/crop%20%26%20tta.ipynb)

but couldn't get to use it in test time augmentation. Any help on how to proceed is appreciated.

# Find the Mixup paper on arXiv and read it. Pick one or two more recent articles introducing variants of Mixup and read them, then try to implement them on your problem.

Read the Mixup paper on [<span class="underline">\[1710.09412\] mixup: Beyond Empirical Risk Minimization</span>](https://arxiv.org/abs/1710.09412)

The first paper [<span class="underline">Bag of Tricks for Image Classification with Convolutional Neural Networks</span>](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) introduces various tricks learned in this chapter and implemented on [<span class="underline">imagenette</span> <span class="underline">dataset</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/practice%20sizing%20and%20tta.ipynb).

The second paper [<span class="underline">\[1907.13037\] Efficient Method for Categorize Animals in the Wild</span>](https://arxiv.org/abs/1907.13037) uses

  -  data augmentation techniques like image transformation, CLAHE (Contrast Adaptive Histogram Equalization) and grayscale

  -  regularization techniques like cutout, mixup and label smoothing to improve generalization of the model.

  -  Ensemble learning is finally used to improve the performance of the model.

Source code [<span class="underline">https://github.com/Walleclipse/iWildCam\_2019\_FGVC6</span>](https://github.com/Walleclipse/iWildCam_2019_FGVC6)

# Find the script training Imagenette using Mixup and use it as an example to build a script for a long training on your own project. Execute it and see if it helps.

Check the source code at [<span class="underline">imagenet mixup.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/imagenet%20mixup.ipynb)

The training and validation loss is quite high in 3 epochs as shown below because the model finds it harder to understand and predict the images. 

![](/images/mixup_result.png)

Training the images for maybe 20 epochs will show a lesser loss. Note the batch size in training is reduced to bs=16. Otherwise the model could not train in my example.

# Read the sidebar "Label Smoothing, the Paper", look at the relevant section of the original paper and see if you can follow it. Don't be afraid to ask for help\!

Label smoothing paper [<span class="underline">\[1512.00567\] Rethinking the Inception Architecture for Computer Vision</span>](https://arxiv.org/abs/1512.00567)

Extract of the core idea below 

![](/images/label%20smoothing%20paper%20extract.png)

#
