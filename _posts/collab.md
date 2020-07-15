# Lesson 8 collab.ipynb

## What problem does collaborative filtering solve?

Collaborative filtering solves recommendation problems. When there are a number of users and a number of products, and you want to recommend which products are most likely to be useful for which users. There are many variations of this: for example, recommending movies (such as on Netflix), figuring out what to highlight for a user on a home page, deciding what stories to show in a social media feed, and so forth.

## How does it solve it?

It solves it using a three step approach

Step 1 is to randomly initialize the latent factors (called parameters in the problem)

Step 2 is to calculate predictions(rating) by taking a dot product of each item(or product like movie) with each user.

Step 3 is to calculate loss(rating predicted - rating actual). We can use any loss function that we wish.

With this in place, we can optimize our parameters (that is, the latent factors) using stochastic gradient descent, such as to minimize the loss. At each step, the stochastic gradient descent optimizer will calculate the match between each movie and each user using the dot product, and will compare it to the actual rating that each user gave to each movie. It will then calculate the derivative of this value(rating predicted - rating actual) and will step the weights by multiplying this by the learning rate. After doing this lots of times, the loss will get better and better, and the recommendations will also get better and better.

## Why might a collaborative filtering predictive model fail to be a very useful recommendation system?

Collaborative filtering predictive models fail to be a very useful recommendation system because of the positive feedback loops. If a small number of users tend to set the direction of your recommendation system, then they are naturally going to end up attracting more users like them to your system. That will amplify the original amplification bias.

## What does a crosstab representation of collaborative filtering data look like?

A crosstab representation is an array of rows and columns where each row is one user and each column is one item. For a movie dataset, the cross tab representation would look like below

![](/images/crosstab%20representation.PNG)

## Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching\!)

Read the code in [<span class="underline">crosstab.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/crosstab.ipynb)

## What is a latent factor? Why is it "latent"?

A latent factor is a factor that is not directly observed but inferred from other observed factors. Latent means hidden.

## What is a dot product? Calculate a dot product manually using pure Python with lists.

Dot product is an element wise multiplication of vectors. Let us say we have two lists

a = \[1,2,3\] and b=\[4,5,6\], the dot product of vectors a and b is \[1\*4, 2\*5, 3\*6\] that is \[4,10,18\]

## What does pandas.DataFrame.merge do?

pandas.DataFrame.merge merges or joins two dataframes with a common column. See [<span class="underline">pandas documentation</span>](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) for examples.

## What is an embedding matrix?

Multiplying by a one-hot-encoded matrix, using the computational shortcut that it can be implemented by simply indexing directly. This is quite a fancy word for a very simple concept. The thing that you multiply the one-hot-encoded matrix by (or, using the computational shortcut, index into directly) is called the *embedding matrix*.

## What is the relationship between an embedding and a matrix of one-hot-encoded vectors?

Embedding = One-hot-encoded vectors \* Embedding Matrix

## Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?

One-hot-encoded vectors are hard coded 0s and 1s values and difficult to read as it is an array representation of the target. For example, a 5 target prediction can be defined as one hot encoded vector \[0 0 0 1 0\] .

Embeddings on the contrary are learnable parameters. By analyzing the relationship between input variables and target, our model can figure out itself the features that seem important.

The model can then update the gradient of loss with respect to the embedding vectors and update these embedding vectors using an optimizer.

## What does an embedding contain before we start training (assuming we're not using a pretained model)?

Embedding contains random numbers as initial parameters before we start training.

## Create a class (without peeking, if possible\!) and use it.

Please see the class and usage in [<span class="underline">class\_creation.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/class_creation.ipynb)

## What does x\[:,0\] return?

X\[:,0\] returns a batches of 64 users picked up from the dataset variable dls

A sample batch looks like this where these numbers are user ids

![](/images/userid.PNG)

## Rewrite the DotProduct class (without peeking, if possible\!) and train a model with it.

Please see [<span class="underline">practice\_collab.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/practice_collab.ipynb)

## What is a good loss function to use for MovieLens? Why?

Mean Squared Error (MSE) loss function is a good loss function to use for MovieLens since it squares (and thereby makes smaller) the difference (or error) between the target and predicted rating. As we see the model starts overfitting very soon (within 5 epochs) hence we need to reduce the gradient of loss function so the model learns slower and thereby avoids overfitting too soon.

## What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?

When we first take the softmax, and then the log likelihood of that, that combination is called *cross-entropy loss*.

Cross-entropy loss predicts the output between 0 and 1. Our output(ratings) are between 1 and 5.

Moreover, cross entropy works on single input whereas here we have two inputs (users and movies). We would have to use Binary Cross-entropy loss instead. Also, we would have to normalize the output to get a rating between 0 and 1.

## What is the use of bias in a dot product model?

Some users are more positive or negative in their recommendations than others, and some movies are just plain better or worse than others. We need to encode these things in our model to make it learn more about input features.

## What is another name for weight decay?

L2 regularization

## Write the equation for weight decay (without peeking\!).

Loss\_with\_decay = loss + weight\_decay \* (parameters \* 2).sum()

## Write the equation for the gradient of weight decay. Why does it help reduce weights?

Parameters.grad + = parameters \* weight decay

Weight decay multiplies weight by the small change close to 0 value (like 0.1 or 0.01) making the weight smaller and thereby gradient of loss smaller.

## Why does reducing weights lead to better generalization?

Limiting the weights from growing too much will make our model slow learner and thereby generalize better.

## What does argsort do in PyTorch?

Argsort sorts the tensor in ascending or descending order by value along the dimension specified.

Example - idxs = movie\_bias.argsort()\[:5\]

## Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?

I don't think so. Movie biases are the outliers or movies that the users don't like(or like) even when the latent factors match their movie choice. If we simply sorted the movies directly by their average rating, it only tells us whether a movie is of a kind that people tend not to enjoy watching. But the movie biases tell that people tend not to like watching it even if it is of a kind that they would otherwise enjoy.

## How do you print the names and details of the layers in a model?

Names and details of the layers in a model can be printed using **learn.model**

## What is the "bootstrapping problem" in collaborative filtering?

Bootstrapping problem is recommending a new user with the very first product/item without having a history of his product likings to learn from. Bootstrapping problem can also be defined from a new product perspective. If we add a new product to the existing portfolio, which users should be recommended with this product given no history of users to learn from.

## How could you deal with the bootstrapping problem for new users? For new movies?

You could assign new

1\. user the *mean of all of the embedding vectors* of your other users

2\. user by picking some particular user to represent *average taste*.

3\. movies by using a tabular model based on user meta data to construct your *initial embedding vector*

## How can feedback loops impact collaborative filtering systems?

A small number of extremely enthusiastic users may end up effectively setting the recommendations for your whole user base. This can form a positive feedback loop where they attract more people of their kind in the recommendation system making the system biased.

Such a divergence can happen too quickly and in a way that it is hidden till it is too late.

## When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?

Using neural networks in collaborative filtering we concatenate the latent factors with users and products respectively and not take the dot product. This gives us a matrix which we can pass through the linear layers and nonlinearities.

If we use unequal size latent factors in dot product model collaborative filtering then the dot product of smaller input with embedding matrix will result in 0s in the remaining values. This will generate a cascading effect when the latent factors are multiplied making more and more values zero in the model.

## Why is there an nn.Sequential in the CollabNN model?

nn.Sequential is used to store nn.Module’s in a cascaded way. In our class CollabNN(Module), we want the network to be in following order - nn.Linear, ReLU, nn.Linear

## What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?

We should use a *tabular model* based on user meta data to construct your initial embedding vector. A model where the dependent variable is a user's embedding vector, and the independent variables are the results of the questions that you ask them, along with their signup metadata.

## Take a look at all the differences between the Embedding version of DotProductBias and the create\_params version, and try to understand why each of those changes is required. If you're not sure, try reverting each change to see what happens. (NB: even the type of brackets used in forward has changed\!)

Code es are highlighted below. For experimentation like change in brackets use the notebook collab.ipynb

\#Embedding version

class DotProductBias(Module):

def \_\_init\_\_(self, n\_users, n\_movies, n\_factors, y\_range=(0,5.5)):

self.user\_factors = **Embedding**(n\_users, n\_factors)

self.user\_bias = Embedding(n\_users, **1**)

self.movie\_factors = Embedding(n\_movies, n\_factors)

self.movie\_bias = Embedding(n\_movies, 1)

self.y\_range = y\_range

def forward(self, x):

users = self.user\_factors**(**x\[:,0\]**)**

movies = self.movie\_factors(x\[:,1\])

res = (users \* movies).sum(dim=1, **keepdim=True**)

res += self.user\_bias**(**x\[:,0\]**)** + self.movie\_bias(x\[:,1\])

return sigmoid\_range(res, \*self.y\_range)

\#create\_params version

class DotProductBias(Module):

def \_\_init\_\_(self, n\_users, n\_movies, n\_factors, y\_range=(0,5.5)):

self.user\_factors = **create\_params**(**\[**n\_users, n\_factors**\]**)

self.user\_bias = create\_params(**\[**n\_users**\]**)

self.movie\_factors = create\_params(\[n\_movies, n\_factors\])

self.movie\_bias = create\_params(\[n\_movies\])

self.y\_range = y\_range

def forward(self, x):

users = self.user\_factors**\[**x\[:,0\]**\]**

movies = self.movie\_factors\[x\[:,1\]\]

res = (users\*movies).sum(dim=1)

res += self.user\_bias**\[**x\[:,0\]**\]** + self.movie\_bias\[x\[:,1\]\]

return sigmoid\_range(res, \*self.y\_range)

## Find three other areas where collaborative filtering is being used, and find out what the pros and cons of this approach are in those areas.

1.mineral exploration,

2.environmental sensing over large areas or multiple sensors;

3.financial data, or

4.electronic commerce

Cons -

  - > Latent factors are not interpretable because there are no content related properties of metadata.

  - > Cold start or bootstrapping. We have no idea on the taste of new users or new item users.

  - > Items that don't get too much data, the model gives them less weight and face popularity bias.

Pros/example application -

We can recommend availability of minerals if we have soil data across the world. For example, we can analyse different soils (user in movielens case) where gold(item in movielens) is available to predict which areas of the world would still have gold left depending on the soil properties.

Historic weather records of places are abundantly available worldwide. One can use this information to predict the future weather or change in environmental conditions in these areas. I need to apply more thought in this area, maybe as a project work.

## Complete this notebook using the full MovieLens dataset, and compare your results to online benchmarks. See if you can improve your accuracy. Look on the book's website and the fast.ai forum for ideas. Note that there are more columns in the full dataset—see if you can use those too (the next chapter might give you ideas).

Full MovieLens dataset has 25 million data samples. I ran out of memory trying to run this notebook. Code solution is available in my notebook [<span class="underline">MovieLens 25M dataset.ipynb</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/MovieLens%2025M%20dataset.ipynb)

## Create a model for MovieLens that works with cross-entropy loss, and compare it to the model in this chapter.

Got an error when I tried to implement (changed y\_range=(0,1))

learn = Learner(dls, model, loss\_func=nn.BCELoss())

learn.fit\_one\_cycle(5, 5e-3, wd=0.01)

RuntimeError: Expected object of scalar type Float but got scalar type Long for argument \#2 'target' in call to \_thnn\_binary\_cross\_entropy\_forward

Since y\_range is a tuple, I couldn't change it to dtype=torch.float(). And if I change y\_range to a tensor the sigmoid\_range() function would not work since it expects a tuple. Hope to learn to modify my error someday or help in fixing this error is appreciated.

## 

## 

##
