# Lesson 06 multicat.ipynb

## How could multi-label classification improve the usability of the bear classifier?

Bear classifier database has images with more than the bear category of objects in many images. Multi-label classification can identify images with more than one match or zero match of objects.

## How do we encode the dependent variable in a multi-label classification problem?

The dependent variable is usually split by a space character so that it becomes a list.

## How do you access the rows and columns of a DataFrame as if it was a matrix?

You can access rows and columns of a DataFrame with the iloc property, as if it were a matrix

## How do you get a column by name from a DataFrame?

You can also get a column by name by indexing into a DataFrame directly

## What is the difference between a Dataset and DataLoader?

A dataset is a *collection* that returns a tuple of our independent and dependent variable for a single item.

A dataLoader is an *iterator* that provides a stream of mini-batches, where each mini-batch is a couple of a batch of independent variables and a batch of dependent variables.

## What does a Datasets object normally contain?

A Dataset object normally contains a train and valid dataset. These are built from a data source like an image or a dataframe.

## What does a DataLoaders object normally contain?

A DataLoader normally contains a training DataLoader and a validation DataLoader.

## What does lambda do in Python?

Lambda is a keyword for defining a function in Python.

Lambda functions are used for quick iterating but are not compatible with serialization like the usual Python functions.

## What are the methods to customize how the independent and dependent variables are created with the data block API?

1\. blocks - helps to define input type like image from eg. ImageBlock and target type like integer from eg. CategoryBlock or Point Block or MultiCategoryBlock

2\. Get\_x - function that gets the input data item

3\. Get\_y - funcion that gets the target data item

4\. Splitter - Function/class that takes the whole data and returns two (or more) list of integers

5\. Item\_tfms - Transforms the input (image) to crop, shrink etc

6\. Summary - method to show what went exactly wrong in building a DataBlock

## Why is softmax not an appropriate output activation function when using a one hot encoded target?

One hot encoding means using a vector of zeros, with a one in each location that is represented in the data, to encode a list of integers. Usually in multi label classification or regression problems, an image can have more than one object/categories as target.

Softmax requires that all predictions sum to 1 and tends to push one activation to be much larger than the others (due to the use of exp); however, we may well have multiple objects that we're confident appear in an image.

Normally for one-hot-encoded targets you'll want F.binary\_cross\_entropy\_with\_logits (or nn.BCEWithLogitsLoss), which do both sigmoid and binary cross-entropy in a single function.

## Why is nll\_loss not an appropriate loss function when using a one-hot-encoded target?

nll\_loss returns the value of just one activation: the single activation corresponding with the single label for an item. This doesn't make sense when we have multiple labels.

## What is the difference between nn.BCELoss and nn.BCEWithLogitsLoss?

nn.BCELoss calculate cross-entropy on a one-hot-encoded target, but do not include the initial sigmoid

nn.BCELogitsLoss does both sigmoid and binary cross-entropy in a single function.

## Why can't we use regular accuracy in a multi-label problem?

Regular accuracy predicted the class with the highest activation(argmax). We can't use it because we could have more than one prediction on a single image.

We need to decide which ones are 0s and which ones are 1s in the prediction by picking a *threshold*. Each value above the threshold will be considered as a 1, and each value lower than the threshold will be considered a 0.

## When is it okay to tune a hyperparameter on the validation set?

In the learned example, the threshold (in accuracy function) is the hyperparameter. Changing the threshold (x-axis) to get a smooth relationship with accuracy (y-axis) is okay to tune on the validation set.

For example, try values between -

*learn.metrics = partial(accuracy\_multi, thresh = 0.1)*

*learn.validate()*

to

*learn.metrics = partial(accuracy\_multi, thresh = 0.99)*

*learn.validate()*

to see what is the best threshold for your dataset. In my bear classifier example, a threshold of \>0.6 works best because there are very few other targets than the bears, person, baby and trees.

## How is y\_range implemented in fastai? (See if you can implement it yourself and test it without peeking\!)

def y\_range(x, hi, lo):

return torch.sigmoid(x)\* (hi-lo) + lo

## What is a regression problem? What loss function should you use for such a problem?

A regression problem is defined by its independent and dependent variables, along with its loss function.

You should use mean squared error loss (MSELoss) for such a problem since coordinates are used as the dependent variable.

## 

## What do you need to do to make sure the fastai library applies the same data augmentation to your inputs images and your target point coordinates?

You should do the same augmentation to the coordinates(dependent variable) as it does to the images (independent variable)

You do this in the DataBlock transforms -

batch\_tfms=\[

\*aug\_transforms(size=(240,320)),

Normalize.from\_stats(\*imagenet\_stats)

\]

## Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.

See my Pandas solutions in repository ‘[<span class="underline">Data-Analysis-with-Python</span>](https://github.com/JayaParmar/Data-Analysis-with-Python)’ namely

1.  > Pandas Average Temperature

2.  > Pandas below zero

3.  > Pandas best record company

4.  > Pandas bicycle timeseries

5.  > Pandas cities

6.  > Pandas commute

7.  > Pandas cycling weather

8.  > Pandas cyclists

9.  > Pandas cyclist per day

10. > Pandas inverse series

11. > Pandas missing value types

12. > Pandas municipal information

13. > Pandas operations on series

14. > Pandas power of series

15. > Pandas snow depth

16. > Pandas special missing values

17. > Pandas split date

18. > Pandas split date continues

19. > Pandas subsetting by positions

20. > Pandas subsetting with loc

21. > Pandas suicide fractions

22. > Pandas suicide weather

23. > Pandas top bands

24. > Python swedish and foreigners

## Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.

See the retrained bear classifier [**<span class="underline">multicat\_bears.ipynb</span>**](https://github.com/JayaParmar/Deep-Learning/blob/master/multicat_bears.ipynb) in my repository ‘[<span class="underline">Deep-Learning</span>](https://github.com/JayaParmar/Deep-Learning)’.

One of the images contained a person but the classifier rightly predicted it as teddy since the person is wearing a teddy bear costume. (Label in first row is target and labels in second row are predictions)

![](/images/person.PNG)

This image had two different kinds of bears (the black bear is not completely visible due to image cropping to 35%). The model has accurately predicted both grizzly and black for an image that was labeled grizzly in the single-label dataset.

![](/images/twobears.PNG)

Another interesting observation is that when the model is not sure of the image, it predicts all labels it is unsure of. In our dataset, labels like book, crown, elephant, cat, light and stick have appeared only once. Hence the model has not got a chance to train sufficiently to learn to identify these labels.

## 

##
