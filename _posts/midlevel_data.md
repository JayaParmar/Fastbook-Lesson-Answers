# Lesson 11 - midlevel\_data

## Why do we say that fastai has a "layered" API? What does it mean?

Fastai has a layered API to give the flexibility of training the models with few lines in the top ‘layer’ API. The mid layer (or mid level) API offers flexibility in case one does not want to use the factory approach.

Mid level API contains functionality for creating

1\. *DataLoaders* for some applications that are not directly supported by fast ai and

2\.*Callback* system to customize the training loop anyway we like.

## Why does a Transform have a decode method? What does it do?

Decode method is used by fastai's show\_batch and show\_results, as well as some other inference methods, to convert predictions and mini-batches into a human-understandable representation.

## Why does a Transform have a setup method? What does it do?

Setup method trains the tokenizer for tokens and creates the vocab for numericalize object.

## How does a Transform work when called on a tuple?

Transform works on a tuple of input and target. It applies the transform to both the input and target separately. You can use Transforms to implement different behavior depending on the type of the input. More information at [<span class="underline">Helper functions for processing data and basic transforms</span>](https://docs.fast.ai/data.transforms)

## Which methods do you need to implement when writing your own Transform?

You need to implement *encodes* method when writing your own Transform. The setups and decodes methods are optional to implement.

## Write a Normalize transform that fully normalizes items (subtract the mean and divide by the standard deviation of the dataset), and that can decode that behavior. Try not to peek\!

@Transform

class Normalize(list):

def setup(self, list): self.mean = sum(list)/len(list)

> def encodes(self,x):

self.stddev = sqrt((x - self.mean)\*\*2)

return (x-self.mean)/self.stddev

def decodes(self,x): return (x+self.mean)\*self.stddev

## Write a Transform that does the numericalization of tokenized texts (it should set its vocab automatically from the dataset seen and have a decode method). Look at the source code of fastai if you need help.

[<span class="underline">https://github.com/fastai/fastai/blob/master/fastai/text/data.py\#L33</span>](https://github.com/fastai/fastai/blob/master/fastai/text/data.py#L33)

## What is a Pipeline?

Pipeline is a class to compose several Transforms together. When you call Pipeline on an object, it will automatically call the transforms inside, in order.

## What is a TfmdLists?

TfmdLists is a class that groups together the Pipeline of Tokenizer & Numericalize with your raw items.

tls = TfmdLists(files, \[Tokenizer.from\_folder(path), Numericalize\])

## What is a Datasets? How is it different from a TfmdLists? 

Dataset applies two (or more) pipelines in parallel to the same raw object and builds a tuple with the result.

x\_tfms = \[Tokenizer.from\_folder(path), Numericalize\]

y\_tfms = \[parent\_label, Categorize()\]

dsets = Datasets(files, \[x\_tfms, y\_tfms\])

TfmdLists is one of the Pipelines which works on the input data. Dataset has a second pipeline which works on the target data (see y\_tfms in the previous question)

## Why are TfmdLists and Datasets named with an "s"?

Both TfmdLists and Datasets work with two sets of data i.e. training set and validation set.

You need to pass a split argument with the indices of elements in each set.

## How can you build a DataLoaders from a TfmdLists or a Datasets?

We can build DataLoaders from Datasets using the dataloaders method. You can pass the batch size and padding type as an argument to this dataloaders method.

## How do you pass item\_tfms and batch\_tfms when building a DataLoaders from a TfmdLists or a Datasets?

Item\_tfms transforms(t.eg. resizes or pads) the input items to the same size. Batch\_tfms transforms a group of items to a batch to be fed to Dataloaders.

Both item\_tfms and batch\_tfms happens when the dataloaders method is called on the Dataset object.

Itef\_tfms happen on the CPU and batch\_tfms can happen on CPU or GPU based on GPU availability. (CPU execution will be slower)

Below is the code to pass these tfms into dataloaders:

item\_tfms **=** \[ToTensor, RandomResizedCrop(128, min\_scale**=**0.35)\]

\#resize the input image to 128 and convert it to a tensor(integer tensor by default)

batch\_tfms **=** \[IntToFloatTensor, Normalize**.**from\_stats(**\***imagenet\_stats)\]

\#transform batch into float tensor and normalizes the batch between 0 & 1 based on imagenet statistics. This can later be fed to GPU where dataloaders is executed as below.

dls **=** dsets**.**dataloaders(after\_item**=**item\_tfms, after\_batch**=**batch\_tfms, bs**=**64, num\_workers**=**8)

## 

## What do you need to do when you want to have your custom items work with methods like show\_batch or show\_results?

1.We need to create an object which calls the setup method.

2.The setup method trains the tokenizer if needed for tok and creates the vocab for num applied to our raw texts (by calling the object as a function).

3.Then finally decode the result back to an understandable representation.

These steps will make the custom items work with methods like show-batch or show\_results.

## Why can we easily apply fastai data augmentation transforms to the SiamesePair we built?

We can easily apply fast ai data augmentation because fastai provides a layered API.

The mid-level API gives you greater flexibility to apply any transformations on your items. In the real-world problems like SiamesePair, this is what we need to make data-munging as easy as possible.

## Use the mid-level API to prepare the data in DataLoaders on your own datasets. Try this with the Pet dataset and the Adult dataset from Chapter 1.

Please refer to [<span class="underline">Pet</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/Siamese%20Pair.ipynb) and [<span class="underline">Adult</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/Adult.ipynb) notebooks in DeepLearning repository for the solution

## Look at the Siamese tutorial in the fastai documentation to learn how to customize the behavior of show\_batch and show\_results for new type of items. Implement it in your own project.

Please refer to [<span class="underline">Siamese Pair</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/Siamese%20Pair.ipynb) notebook in the Deep Learning repository.
