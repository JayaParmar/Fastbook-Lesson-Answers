# Lesson 5 ‘pet\_breeds.ipynb’

## Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?

We first resize the items in an image dataset to an equal large size item so that the images become equal size. Then we make batches of small images from this large image and send it to the GPU so that the GPU can focus on finding important features from every small image.

If the dataset is not resized to the same large size, the model will not train.

## If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.

Please refer to my regular expressions exercises here

[<span class="underline">https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20File%20listing</span>](https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20File%20listing)

[<span class="underline">https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20word%20frequencies</span>](https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20word%20frequencies)

[<span class="underline">https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20red%20green%20blue</span>](https://github.com/JayaParmar/Data-Analysis-with-Python/blob/master/Part%202%20red%20green%20blue)

## What are the two ways in which data is most commonly provided, for most deep learning datasets?

  - > Individual files representing items of data, such as text documents or images, possibly organized into folders or with filenames representing information about those items

  - > A table of data, such as in CSV format, where each row is an item which may include filenames providing a connection between the data in the table and data in other formats, such as text documents and images

## Look up the documentation for L and try using a few of the new methods that it adds.

Init signature: L(items=None, \*rest, use\_list=False, match=None)

Source:

class L(CollBase, metaclass=NewChkMeta):

"Behaves like a list of \`items\` but can also index with list of indices or masks"

\_default='items'

def \_\_init\_\_(self, items=None, \*rest, use\_list=False, match=None):

if rest: items = (items,)+rest

if items is None: items = \[\]

if (use\_list is not None) or not \_is\_array(items):

items = list(items) if use\_list else \_listify(items)

if match is not None:

if is\_coll(match): match = len(match)

if len(items)==1: items = items\*match

else: assert len(items)==match, 'Match length mismatch'

super().\_\_init\_\_(items)

@property

def \_xtra(self): return None

def \_new(self, items, \*args, \*\*kwargs): return type(self)(items, \*args, use\_list=None, \*\*kwargs)

def \_\_getitem\_\_(self, idx): return self.\_get(idx) if is\_indexer(idx) else L(self.\_get(idx), use\_list=None)

def copy(self): return self.\_new(self.items.copy())

def \_get(self, i):

if is\_indexer(i) or isinstance(i,slice): return getattr(self.items,'iloc',self.items)\[i\]

i = mask2idxs(i)

return (self.items.iloc\[list(i)\] if hasattr(self.items,'iloc')

else self.items.\_\_array\_\_()\[(i,)\] if hasattr(self.items,'\_\_array\_\_')

else \[self.items\[i\_\] for i\_ in i\])

def \_\_setitem\_\_(self, idx, o):

"Set \`idx\` (can be list of indices, or mask, or int) items to \`o\` (which is broadcast if not iterable)"

if isinstance(idx, int): self.items\[idx\] = o

else:

idx = idx if isinstance(idx,L) else \_listify(idx)

if not is\_iter(o): o = \[o\]\*len(idx)

for i,o\_ in zip(idx,o): self.items\[i\] = o\_

def \_\_iter\_\_(self): return iter(self.items.itertuples() if hasattr(self.items,'iloc') else self.items)

def \_\_contains\_\_(self,b): return b in self.items

def \_\_invert\_\_(self): return self.\_new(not i for i in self)

def \_\_eq\_\_(self,b): return False if isinstance(b, (str,dict,set)) else all\_equal(b,self)

def \_\_repr\_\_(self): return repr(self.items) if \_is\_array(self.items) else coll\_repr(self)

def \_\_mul\_\_ (a,b): return a.\_new(a.items\*b)

def \_\_add\_\_ (a,b): return a.\_new(a.items+\_listify(b))

def \_\_radd\_\_(a,b): return a.\_new(b)+a

def \_\_addi\_\_(a,b):

a.items += list(b)

return a

def sorted(self, key=None, reverse=False):

if isinstance(key,str): k=lambda o:getattr(o,key,0)

elif isinstance(key,int): k=itemgetter(key)

else: k=key

return self.\_new(sorted(self.items, key=k, reverse=reverse))

@classmethod

def split(cls, s, sep=None, maxsplit=-1): return cls(s.split(sep,maxsplit))

@classmethod

def range(cls, a, b=None, step=None):

if is\_coll(a): a = len(a)

return cls(range(a,b,step) if step is not None else range(a,b) if b is not None else range(a))

def map(self, f, \*args, \*\*kwargs):

g = (bind(f,\*args,\*\*kwargs) if callable(f)

else f.format if isinstance(f,str)

else f.\_\_getitem\_\_)

return self.\_new(map(g, self))

def filter(self, f, negate=False, \*\*kwargs):

if kwargs: f = partial(f,\*\*kwargs)

if negate: f = negate\_func(f)

return self.\_new(filter(f, self))

def argwhere(self, f, negate=False, \*\*kwargs):

if kwargs: f = partial(f,\*\*kwargs)

if negate: f = negate\_func(f)

return self.\_new(i for i,o in enumerate(self) if f(o))

def unique(self): return L(dict.fromkeys(self).keys())

def enumerate(self): return L(enumerate(self))

def val2idx(self): return {v:k for k,v in self.enumerate()}

def itemgot(self, \*idxs):

x = self

for idx in idxs: x = x.map(itemgetter(idx))

return x

def attrgot(self, k, default=None): return self.map(lambda o:getattr(o,k,default))

def cycle(self): return cycle(self)

def map\_dict(self, f=noop, \*args, \*\*kwargs): return {k:f(k, \*args,\*\*kwargs) for k in self}

def starmap(self, f, \*args, \*\*kwargs): return self.\_new(itertools.starmap(partial(f,\*args,\*\*kwargs), self))

def zip(self, cycled=False): return self.\_new((zip\_cycle if cycled else zip)(\*self))

def zipwith(self, \*rest, cycled=False): return self.\_new(\[self, \*rest\]).zip(cycled=cycled)

def map\_zip(self, f, \*args, cycled=False, \*\*kwargs): return self.zip(cycled=cycled).starmap(f, \*args, \*\*kwargs)

def map\_zipwith(self, f, \*rest, cycled=False, \*\*kwargs): return self.zipwith(\*rest, cycled=cycled).starmap(f, \*\*kwargs)

def concat(self): return self.\_new(itertools.chain.from\_iterable(self.map(L)))

def shuffle(self):

it = copy(self.items)

random.shuffle(it)

return self.\_new(it)

def append(self,o): return self.items.append(o)

def remove(self,o): return self.items.remove(o)

def count (self,o): return self.items.count(o)

def reverse(self ): return self.items.reverse()

def pop(self,o=-1): return self.items.pop(o)

def clear(self ): return self.items.clear()

def index(self, value, start=0, stop=sys.maxsize): return self.items.index(value, start, stop)

def sort(self, key=None, reverse=False): return self.items.sort(key=key, reverse=reverse)

def reduce(self, f, initial=None): return reduce(f, self) if initial is None else reduce(f, self, initial)

def sum(self): return self.reduce(operator.add)

def product(self): return self.reduce(operator.mul)

File: /opt/conda/lib/python3.7/site-packages/fastcore/foundation.py

Type: NewChkMeta

Subclasses: TfmdLists, MultiCategory, LabeledBBox

## Look up the documentation for the Python pathlib module and try using a few methods of the Path class.

Example - Here I make a path called dataset and download the zip file images there

path = Path('dataset')

if not path.exists():

path.mkdir()

with zipfile.ZipFile("flowers-recognition.zip","r") as zip\_ref: zip\_ref.extractall(path)

## Give two examples of ways that image transformations can degrade the quality of the data.

Image transformation might introduce a spurious empty zone or create reflection. For example, in the bottom image the image on the right is less well defined and has reflection padding artifacts in the bottom-left corner; also, the grass at the top left has disappeared entirely.

![](name//media/image1.png)

## What method does fastai provide to view the data in a DataLoaders?

To get a batch of real data from our DataLoaders, we can use the one\_batch method

## What method does fastai provide to help you debug a DataBlock?

Summary method used as variable.summary.path()

## Should you hold off on training a model until you have thoroughly cleaned your data?

No, one shouldn't hold off on training a model until the data is thoroughly cleaned.

## What are the two pieces that are combined into cross-entropy loss in PyTorch?

Softmax is the first part of the cross-entropy loss, the second part is log likelihood.

## What are the two properties of activations that softmax ensures? Why is this important?

1.  > Our activations (or probability of categories), after softmax, are between 0 and 1, and sum to 1 for each row in the batch of predictions.

2.  > Works with more than just binary classification—it classifies any number of categories.

Softmax activations are then fed into log likelihood to find the loss. Values between 0 to 1 get amplified by taking log i.e. small values become very small and large values become closer to 1. That is why this is important.

## When might you want your activations to not have these two properties?

While doing binary classification. Our activations should be 0 or 1.

## Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?

torch.where is used to select between inputs and 1-inputs. It wouldn't show high differences between categories due to its subtraction from one. In order to make small values smaller and large values larger we need to take log into the loss function.

## What is the value of log(-2)? Why?

There is no solution to the log of negative numbers. Watch this to understand why youtube.com/watch?v=MuX7T4PM1Mc

## What are two good rules of thumb for picking a learning rate from the learning rate finder?

  - > One order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10)

  - > The last point where the loss was clearly decreasing

## What two steps does the fine\_tune method do?

  - > Trains the randomly added layers for one epoch, with all other layers frozen

  - > Unfreezes all of the layers, and trains them all for the number of epochs requested

## In Jupyter Notebook, how do you get the source code for a method or function?

By typing ?? before the function and after the method ??

## What are discriminative learning rates?

We need not use the optimum learning rate we got with the learning rate finder method for the deeper layers. This is because the pre-trained model has been trained for hundreds of epochs and millions of images. Thus we can use a higher learning rate for the deeper layers. Usage of two (or more) different learning rates is called discriminative learning rates.

## How is a Python slice object interpreted when passed as a learning rate to fastai?

The first value passed will be the learning rate in the **earliest layer** of the neural network, and the second value will be the learning rate in the **final layer**.

## Why is early stopping a poor choice when using 1cycle training?

Because those epochs in the middle occur before the learning rate has had a chance to reach the small values, where it can really find the best result.

## What is the difference between resnet50 and resnet101?

50 and 101 are the layers in the resnet architecture. A 101 ( layers and parameters; sometimes described as the "capacity" of a model) version of a ResNet will always be able to give us a better training loss, but it can suffer more from overfitting, because it has more parameters to overfit with.

## What does to\_fp16 do?

One technique that can speed things up a lot is *mixed-precision training*. This refers to using less-precise numbers (*half-precision floating point*, also called *fp16*) where possible during training. NVIDIA GPUs support a special feature called *tensor cores* that can dramatically speed up neural network training, by 2-3x. They also require a lot less GPU memory. To enable this feature in fastai, just add to\_fp16() after your Learner creation (you also need to import the module).

**from** **fastai2.callback.fp16** **import** \*

## See if you can improve the accuracy of the classifier in this chapter. What's the best accuracy you can achieve? 

My best accuracy achieved is 90.9337 (loss=0.909337‬)

See [<span class="underline">https://github.com/JayaParmar/Deep-Learning/blob/master/practice\_pet\_breeds.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/practice_pet_breeds.ipynb)
