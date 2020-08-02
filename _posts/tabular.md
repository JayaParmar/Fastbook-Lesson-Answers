## What is a continuous variable?

In tabular data some columns may contain numerical data which can be directly fed to the model, for example age. This is a continuous variable.

## What is a categorical variable?

In tabular data some columns are non-numerical or categories, for example sex, which need to be converted to numbers before feeding to the model. This is a categorical variable.

## Provide two of the words that are used for the possible values of a categorical variable.

Sex, movieID

## What is a "dense layer"?

The raw categorical data is transformed by an embedding layer before it interacts with the raw continuous input data. This data is then concatenated and fed into a dense layer for the model to learn. Dense layer is the layer where all data is treated as continuous data irrespective of its variable type. Moreover, it is a linear layer which the model already knows how to train on.

## How do entity embeddings reduce memory usage and speed up neural networks?

Entity embedding treats or maps every category to a number. Unlike one hot encoding which is stored as an array of 0s and 1s, entity embedding is stored as a single number which reduces memory usage. The mapping is fed to the neural network which learns it during the standard supervised training process. This speeds up the neural network learning compared to one hot encoding because it is easier to learn from a continuous variable like number than an array of massive 0s and 1s.

## What kinds of datasets are entity embeddings especially useful for?

Entity embeddings are useful for high cardinality variables where the values of different categorical variables are quite similar to each other. Models tend to overfit in such cases. Entity embedding would place similar variables near to each other in the Euclidean space making it easier to visualize the relationship between them.

This will later help to eliminate certain categorical variables (or features) to simplify the model and thereby understand the deeper relationship of the target with the more important input categories/features.

## What are the two main families of machine learning algorithms?

1.  Ensembles of decision trees (i.e., random forests and gradient boosting machines), mainly for structured data (such as you might find in a database table at most companies)

2.  Multilayered neural networks learned with SGD (i.e., shallow and/or deep learning), mainly for unstructured data (such as audio, images, and natural language)

## Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?

Some categorical columns (ordinal columns) have a natural order which needs to be explained to the entity embedding since it has a meaning. These columns are generally strings. For example, a column Product Size is an array of \[nan, 'Medium', 'Small', 'Large / Medium', 'Mini', 'Large', 'Compact'\].

We need to set these categories in an order before embedding them. In Pandas it is done like below:

sizes = \['Large', 'Large / Medium', 'Medium', 'Small', 'Mini', 'Compact'\]

df\[‘ProductSize’\] = df.\[‘ProductSize’\].astype(category)

df\[‘ProductSize’\].cat.set\_categories(sizes, ordered = True, inplace=True)

## Summarize what a decision tree algorithm does.

1.  > Loop through each column of the dataset in turn.

2.  > For each column, loop through each possible level of that column in turn.

3.  > Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable).

4.  > Find the average dependent variable (sale price in our case) for each of those two groups, and see how close that is to the actual dependent variable of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average sale price of the item's group.

5.  > After looping through all of the columns and all the possible levels for each, pick the split point that gave the best predictions using that simple model.

6.  > We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each by going back to step 1 for each group.

7.  > Continue this process recursively, until you have reached some stopping criterion for each group—for instance, stop splitting a group further when it has only 20 items in it.

## Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow it to be used in a model?

Dates are qualitatively different from others in a way that is often relevant to the systems we are modeling.

We want to predict the dependent variable for the future based on the information provided from the past.i.e. the data is divided based on time. Hence date has a special significance in our entire modelling. Moreover, the relationship of dependent variables with the independent variables which was important in the past might not be equally important in the present. Hence we want to take the most recent relationship to predict the future. Date will help us split the dataset.

We preprocess dates using a fastai function ‘add\_datepart’ which splits the date information into year, month, week, day, dayofweek, dayofyear, is month end, is month start, is quarter end, is quarter start, is year end, is year start, time elapsed like multiple columns.

## Should you pick a random validation set in the bulldozer competition? If not, what kind of validation set should you pick?

No you should not pick a random validation set in the bulldozer competition because the dataset is time-series information. Validation set should be later in time than the training set. We pick the most recent date information in the validation set. If 20% is the validation set then we pick the most recent 20% dates based dataset to make our validation set.

## What is pickle and what is it useful for?

Pickle is a Python system to save any Python object. It is useful to save the data one has worked on to avoid rework in future.

## How are mse, samples, and values calculated in the decision tree drawn in this chapter?

def r\_mse(pred,y): return round(math.sqrt(((pred-y)\*\*2).mean()),6)

In the training dataset (TrainAndValid.csv) average of sale price column is 31 215. This is 412 698 *samples*. Log base e for this *value* is 10.34

*Mse* is the mean squared error. Decision tree chooses to split on the coupler system as the easiest split column. Coupler system has categories yes; no or unspecified; blank. After embedding these categories, there are 404 710 rows or *samples* to look at. For all these samples, it takes a log of average sale price which is the *value* 10.1. It takes a difference of this value from the original value which was 10.34 and finds the square of this difference and later means it.That is *mse* 0.48 in our picture.

The decision tree then splits on a coupler system embedding value less than or equal to 0.5 and proceeds to look at YearMade column which it finds second best to split on and repeats the above process.

## How do we handle categorical variables in a decision tree?

Categorical variables are embedded before feeding into the decision tree. The tree then splits the data into two groups based on whether they are equal to or not equal to that level of that categorical variable.

## What is bagging?

Bagging is a method for generating multiple versions of a predictor and using these to get an aggregated predictor. The procedure is as below:

1.  > Randomly choose a subset of the rows of your data (i.e., "bootstrap replicates of your learning set").

2.  > Train a model using this subset.

3.  > Save that model, and then return to step 1 a few times.

4.  > This will give you a number of trained models. To make a prediction, predict using all of the models, and then take the average of each of those model's predictions.

## What is the difference between max\_samples and max\_features when creating a random forest?

max\_samples defines how many rows to sample for training each tree, and max\_features defines how many columns to sample at each split point (where 0.5 means "take half the total number of columns").

## If you increase n\_estimators to a very high value, can that lead to overfitting? Why or why not?

n\_estimators define the number of trees we want. If we increase the trees to a very high value it cannot lead to overfitting since each tree learns to predict its own value. It does not know what the other tree is predicting. This will improve the accuracy of the model.

## What is "out-of-bag-error"?

The OOB error is a way of measuring prediction error on the training set by only including in the calculation of a row's error trees where that row was *not* included in training. This allows us to see whether the model is overfitting, without needing a separate validation set.

## Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses?

OOB error is predicted on a ‘thrown away’ sample from a bag or training set. In other words, the sample which does not generalize very well in a given bag. Since the model has already seen this sample, it is easy to ‘accommodate’ it in the generalization to reduce the error.

In a time-series dataset, the validation data includes later dates i.e. the model has not yet seen the features related to later dates (and thereby learnt) to predict on. OOB sample is within the dates shown to the model and now the model should learn to generalize more broadly.

Hypotheses checking around here -

**def** get\_oob(df):

m = RandomForestRegressor(n\_estimators=40, min\_samples\_leaf=15,

max\_samples=50000, max\_features=0.5, n\_jobs=-1, oob\_score=**True**)

m.fit(df, y)

**return** m.oob\_score\_

xs\_final = xs\_imp.drop(to\_drop, axis=1)

valid\_xs\_final = valid\_xs\_imp.drop(to\_drop, axis=1)

## Explain why random forests are well suited to answering each of the following question:

## How confident are we in our predictions using a particular row of data?

We can know the confidence of the predictions by using the standard deviation of predictions across the trees, instead of just the mean. This tells us the *relative* confidence of predictions.

We would want to be more cautious of using the results for rows where trees give very different results (higher standard deviations), compared to cases where they are more consistent (lower standard deviations).

## For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?

We take that one row of data and put it through the first decision tree, looking to see what split is used at each point throughout the tree (for instance, see the waterfall chart prediction for the first row below). For each split, we see what the increase or decrease in the addition is, compared to the parent node of the tree. We do this for every tree, and add up the total change in importance by the split variable.

![](/images/waterfall.png)

## Which columns are the strongest predictors?

We can know the strongest predictors columns by looking at the *feature importance.* We can get these directly from sklearn's random forest by looking in the feature\_importances\_ attribute.

## How do predictions vary as we vary these columns?

Predictions dont change if we remove the columns with less feature\_importance. We could use just a subset of the columns by removing the variables of low importance and still get good results. For instance, start with just keeping columns with a feature importance greater than 0.005.

## What's the purpose of removing unimportant variables?

Studying too many variables in depth can be difficult. We simplify the model which makes it more interpretable and easier to roll out and maintain.

## What's a good type of plot for showing tree interpreter results?

A *waterfall plot* is a good type of plot for showing tree interpreter results. It shows how the positive and negative contributions from all the independent variables sum up to create the final prediction, which is the right hand column labeled "net" in the chart above.

## What is the "extrapolation problem"?

Extrapolation problem is the inability to generalize well to new data.

A random forest just averages the predictions of a number of trees. And a tree simply predicts the average value of the rows in a leaf. Therefore, a tree and a random forest can never predict values outside of the range of the training data.

For example, data where there is a trend over time, such as inflation, and you wish to make predictions for a future time, your predictions will be systematically too low.

We need to make sure our validation set does not contain out-of-domain data.

## How can you tell if your test or validation set is distributed in a different way than your training set?

We use the random forest to predict whether a row is in the validation set or the training set. To see this in action, we can combine our training and validation sets together, create a dependent variable that represents which dataset each row comes from, build a random forest using that data, and get its feature importance.

## Why do we make saleElapsed a continuous variable, even though it has less than 9,000 distinct values?

saleElapsed is the number of days between the start of the dataset and each row, so it directly encodes the date. We want to predict the sale price on a future date with respect to this variable. The model should understand that it needs to take the difference between the training dataset start date and each date given in the validation set row. This can be done by treating saleElapsed as a number. i.e. continuous variable. A categorical variable cannot, by definition, extrapolate outside the range of values that it has seen.

## What is "boosting"?

Boosting is an ensemble method where we **add** models instead of averaging them (which is done in bagging)

Boosting works as below:

  - > Train a small model that underfits your dataset.

  - > Calculate the predictions in the training set for this model.

  - > Subtract the predictions from the targets; these are called the "residuals" and represent the error for each point in the training set.

  - > Go back to step 1, but instead of using the original targets, use the residuals as the targets for the training.

  - > Continue doing this until you reach some stopping criterion, such as a maximum number of trees, or you observe your validation set error getting worse.

## How could we use embeddings with a random forest? Would we expect this to help?

The embeddings obtained from the trained neural network boost the performance of all tested machine learning methods considerably when used as the input features instead. This would bring down the error in all machine learning methods including random forest.

You could just use an embedding, which is literally just an array lookup, along with a small decision tree ensemble.

## Why might we not always use a neural net for tabular modeling?

Neural networks take the

  - > longest time to train, and

  - > require extra preprocessing, such as normalization;

  - > this normalization needs to be used at inference time as well.

## Pick a competition on Kaggle with tabular data (current or past) and try to adapt the techniques seen in this chapter to get the best possible results. Compare your results to the private leaderboard.

Please see the competition code in [<span class="underline">Supply Chain Shipment Price Data Analysis</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/Supply%20Chain%20Shipment%20Pricing%20Data.ipynb)

## Implement the decision tree algorithm in this chapter from scratch yourself, and try it on the dataset you used in the first exercise.

In the above notebook, refer to the section ‘Decision Tree Model’ for details.

## Use the embeddings from the neural net in this chapter in a random forest, and see if you can improve on the random forest results we saw.

In the above notebook, refer to the section ‘Neural Networks Model’ there accuracy after neural network embeddings is compared with the random forest results.

## Explain what each line of the source of TabularModel does (with the exception of the BatchNorm1d and Dropout layers).

## class TabularModel

TabularModel(emb\_szs:ListSizes, n\_cont:int, out\_sz:int, layers:Collection\[int\], ps:Collection\[float\]=None, emb\_drop:float=0.0, y\_range:OptRange=None, use\_bn:bool=True, bn\_final:bool=False) ::

Basic model for tabular data.

emb\_szs match each categorical variable size with an embedding size, n\_cont is the number of continuous variables. The model consists of Embedding layers for the categorical variables, followed by a Dropout of emb\_drop, and a BatchNorm for the continuous variables. The results are concatenated and followed by blocks of BatchNorm, Dropout, Linear and ReLU (the first block skips BatchNorm and Dropout, the last block skips the ReLU).

The sizes of the blocks are given in [layers](https://docs.fast.ai/layers.html#layers) and the probabilities of the Dropout in ps. The last size is out\_sz, and we add a last activation that is a sigmoid rescaled to cover y\_range (if it's not None). Lastly, if use\_bn is set to False, all BatchNorm layers are skipped except the one applied to the continuous variables.

Generally it's easiest to just create a learner with [tabular\_learner](https://docs.fast.ai/tabular.learner.html#tabular_learner), which will automatically create a [TabularModel](https://docs.fast.ai/tabular.models.html#TabularModel) for you.

In \[ \]:

## 

## 

## 

## 

## 

##
