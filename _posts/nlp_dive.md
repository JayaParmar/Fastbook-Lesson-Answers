# Lesson 12 - nlp\_dive.ipynb

## If the dataset for your project is so big and complicated that working with it takes a significant amount of time, what should you do?

You should try to start with a simpler dataset and test your methods on it first before moving to the big and complicated ones. If needed, you can make a simplest dataset on your own.

## Why do we concatenate the documents in our dataset before creating a language model?

We need to create tokens for all the words in our dataset. If we leave out the validation set, our vocabulary will not capture those words. Hence we concatenate the documents and make tokens for every word. Later we can split the documents again into 80% train and 20% validation set before training the language model.

## To use a standard fully connected network to predict the fourth word given the previous three words, what two tweaks do we need to make to our model?

The first tweak is that the first linear layer will use only the first word's embedding as activations, the second layer will use the second word's <span class="underline">embedding</span> plus the first layer's <span class="underline">output activations</span>, and the third layer will use the third word's embedding plus the second layer's output activations. The key effect of this is that every word is interpreted in the information context of any words preceding it.

The second tweak is that each of these three layers will use the <span class="underline">same weight</span> matrix. The way that one word <span class="underline">impacts</span> the activations from previous words should not change depending on the position of a word. In other words, activation values will change as data moves through the layers, but the layer weights themselves will not change from layer to layer. So, a layer does not learn one sequence position; it must learn to handle all positions.

## How can we share a weight matrix across multiple layers in PyTorch?

We can create one layer and use it multiple times. While defining the PyTorch class, we initialize the weight matrix with the input, hidden and output layers. Then we use the same weight matrix on every new input via the initialized variables.

## Write a module that predicts the third word given the previous two words of a sentence, without peeking.

class LMModel(Module):

def \_\_init\_\_(self, vocab\_sz\_, n\_hidden):

self.i\_h = nn.Embedding(vocab\_sz, n\_hidden)

self.h\_h = nn.Linear(n\_hidden, n\_hidden)

self.o\_h = nn.Linear(n\_hidden, vocab\_sz)

def forward(self, x):

h=0

for i in range(2):

h=h+self.i\_h(x\[:,i\])

h = F.relu(self.h\_h(h))

out = self.o\_h(h)

## What is a recurrent neural network?

A neural network that reoccurs. A neural network that is defined using a loop (like above example) is a recurrent neural network.

## What is "hidden state"?

The activations that are updated at each step of the recurrent neural network (or loop) are hidden state/s.

## What is the equivalent of hidden state in LMModel1?

The activation generated after ReLU (Rectified Linear Unit) function is the hidden state in LMModel1. It is stored in variable h.

## To maintain the state in an RNN, why is it important to pass the text to the model in order?

If we order the text to the model in order, those text sequences will be read in order by the model, exposing the model to long stretches of the original sequence.

## What is an "unrolled" representation of an RNN?

![](name//media/image2.png)

## Why can maintaining the hidden state in an RNN lead to memory and performance problems? How do we fix this problem?

Maintaining the hidden state will make the RNN calculate the derivative of all hidden layers and backpropagate them all the way to the first layer. This will make the network very slow indeed, and very memory-intensive.

The solution to this problem is to tell PyTorch that we do not want to back propagate the derivatives through the entire implicit neural network. Instead, we will just keep the last three layers of gradients. To remove all of the gradient history in PyTorch, we use the detach method (self.h.detach=0).

## What is "BPTT"?

Backpropagation through time (BPTT): Treating a neural net with effectively one layer per time step (usually refactored using a loop) as one big model, and calculating gradients on it in the usual way. To avoid running out of memory and time, we usually use *truncated* BPTT, which "detaches" the history of computation steps in the hidden state <span class="underline">every few time steps.</span>

## Write code to print out the first few batches of the validation set, including converting the token IDs back into English strings, as we showed for batches of IMDb data

seqs = L((tensor(nums\[i:i+sl\]), tensor(nums\[i+1:i+sl+1\]))

**for** i **in** range(0,len(nums)-sl-1,sl))

cut = int(len(seqs) \* 0.8)

dls = DataLoaders.from\_dsets(group\_chunks(seqs\[:cut\], bs),

group\_chunks(seqs\[cut:\], bs),

bs=bs, drop\_last=**True**, shuffle=**False**)

\[L(vocab\[o\] **for** o **in** s) **for** s **in** seqs\[0\]\]

What does the ModelResetter callback do? Why do we need it?

ModelResetter callback resets the model after each training and validation step. The model should ‘forget’ what it learned in the past and start afresh on a new set of words.

## What are the downsides of predicting just one output word for each three input words?

The downside is that we are not feeding back is not as large as it could be. It would be better if we predicted the next word after every single word, rather than every three words.

## Why do we need a custom loss function for LMModel4?

LMModel4 will return outputs of shape bs x sl x vocab\_sz (since we stacked on dim=1). Our targets are of shape bs x sl, so we need to flatten those before using them in F.cross\_entropy.

## Why is the training of LMModel4 unstable?

LMModel4 is unstable while training because we have a very deep network which can result in very large or very small gradients.

## In the unrolled representation, we can see that a recurrent neural network actually has many layers. So why do we need to stack RNNs to get better results?

We stack to save a sequence of previous hidden layers which will be used as input to the subsequent RNN layer. This representation helps the RNN learn from the past inputs.

## Draw a representation of a stacked (multilayer) RNN.

![](name//media/image1.png)

## Why should we get better results in an RNN if we call detach less often? Why might this not happen in practice with a simple RNN?

If we detach less often, the model will have more layers, giving our RNN a longer time horizon to learn from, and richer features to create. In practice, a simple RNN has only one linear layer between the hidden state and the output activations.

## Why can a deep network result in very large or very small activations? Why does this matter?

Deep network is layers of matrix multiplications. A weight multiplied by previous activation is the next activation.

For example, if you multiply by 2, starting at 1, you get the sequence 1, 2, 4, 8,... after 32 steps you are already at 4,294,967,296. A similar issue happens if you multiply by 0.5: you get 0.5, 0.25, 0.125… and after 32 steps it's 0.00000000023. As you can see, multiplying by a number even slightly higher or lower than 1 results in an explosion or disappearance of our starting number, after just a few repeated multiplications.

This matters because computers store numbers as "floating point" and they become less and less accurate the further away the numbers get from zero.

## In a computer's floating-point representation of numbers, which numbers are the most precise?

Floating point numbers represented with 32 bits (single precision) or 64 bits (double precision) are most precise.

## Why do vanishing gradients prevent training?

Vanishing gradients make the derivatives extremely small meaning the activations become little and model stops to improve. Now even if we train the model for many more times, the performance will increase negligibly.

## Why does it help to have two hidden states in the LSTM architecture? What is the purpose of each one?

It helps to keep two hidden states as one state keeps the memory of the previous hidden state and the second can focus on predicting the output. The memory keeping state is called the cell state (ct) and the token predicting state is the input state (ht-1).

## What are these two states called in an LSTM? 

Cell state and input state

## What is tanh, and how is it related to sigmoid?

Tanh is an activation function. IT is a sigmoid function rescaled to the range -1 to 1.

## What is the purpose of this code in LSTMCell: 

## h = torch.stack(\[h, input\], dim=1)

It will stack the input and the two hidden states(cell and input) along the dimension 1.

This output will go through the sigmoid function (neural network) to generate an activation.

## What does chunk do in PyTorch?

Chunk will split the tensor into number of pieces mentioned, t.chunk(n)

## Study the refactored version of LSTMCell carefully to ensure you understand how and why it does the same thing as the non-refactored version.

It rewrites the matrix multiplication code for 4 individual gates to one multiplication using the chunk method with 4 tensor pieces.

## Why can we use a higher learning rate for LMModel6?

LMModel6 is a two layer LSTM model. It removes the exploding and vanishing gradient problem of RNN thereby avoiding activations becoming too high or too low. A higher learning rate will make the model learn faster and improve the accuracy faster.

## What are the three regularization techniques used in an AWD-LSTM model?

Dropout, activation regularization, and temporal activation regularization.

## What is "dropout"?

Dropout is a technique to randomly change some activations to zero at training time. This makes sure all neurons actively work toward the output

## Why do we scale the weights with dropout? Is this applied during training, inference, or both?

The weights are scaled with dropout to make the neurons cooperate better together. This makes the activations more noisy, thus making the model more robust. This is applied during training and not inference.

## What is the purpose of this line from Dropout: if not self.training: return x

This line is to avoid dropout during the inference/validation set. The dropout should happen only during the training.

## Experiment with bernoulli\_ to understand how it works.

Bernoulli\_ gives dropout a probability between 0 and 1. (1-p) is the probability of dropping a node from the network. If p = 0.5 then (1-p)=0.5 which means there is a 50% probability that the node will be dropped from the network (during training).

mask = x.new(\*x.shape).bernoulli\_(1-p)

**return** x \* mask.div\_(1-p)

## How do you set your model in training mode in PyTorch? In evaluation mode?

By default the ode is initialised to training mode i.e. self.training= True. When one wants to set the model to an evaluation model, self.training=False need to be set.

Alternatively, you can set it as model.train() or model.eval() also.

## Write the equation for activation regularization (in math or code, as you prefer). How is it different from weight decay?

loss + = alpha \* activation \* pow(2) \* mean()

The multiplier alpha is similar to weight decay

## Write the equation for temporal activation regularization (in math or code, as you prefer). Why wouldn't we use this for computer vision problems?

loss + = beta \* (activations\[:,1:\] - activations\[:,:-1\])\*powe(2)\*mean()

TAR penalizes any large changes in hidden state between timestamps, encouraging the model to keep the output as consistent as possible. Note that it takes the difference of activations between two dimensions thereby making the loss small (and output consistent). Computer vision problems need to identify different images where the model cannot keep the output consistent.

## What is "weight tying" in a language model?

Weight tying shares the weight between embedding and <span class="underline">softmax layer</span>, substantially reducing the total parameter count in the model.

In a language model, the input embeddings represent a mapping from English words to activations, and the <span class="underline">output hidden layer</span> represents a mapping from activations to English words. We might expect, intuitively, that these mappings could be the same.

self.h\_o.weight = self.i\_h.weight

## In LMModel2, why can forward start with h=0? Why don't we need to say h=torch.zeros(...)?

I don't see much change by replacing h=0 with h=torch.zeros(64)

See code below for the accuracy in first and second case

class LMModel2(Module):

def \_\_init\_\_(self, vocab\_sz,n\_hidden):

self.i\_h = nn.Embedding(vocab\_sz, n\_hidden)

self.h\_h = nn.Linear(n\_hidden, n\_hidden)

self.h\_o = nn.Linear(n\_hidden, vocab\_sz)

def forward(self,x):

**h = 0**

for i in range(3):

h = h + self.i\_h(x\[:,i\])

h = F.relu(self.h\_h(h))

return self.h\_o(h)

| **epoch** | **train\_loss** | **valid\_loss** | **accuracy** | **time** |
| --------- | --------------- | --------------- | ------------ | -------- |
| 0         | 1.816274        | 1.964143        | 0.460185     | 00:02    |
| 1         | 1.423805        | 1.739964        | 0.473259     | 00:02    |
| 2         | 1.430327        | 1.685172        | 0.485382     | 00:02    |
| 3         | 1.388390        | 1.657033        | 0.470406     | 00:02    |

class LMModel2(Module):

def \_\_init\_\_(self, vocab\_sz,n\_hidden):

self.i\_h = nn.Embedding(vocab\_sz, n\_hidden)

self.h\_h = nn.Linear(n\_hidden, n\_hidden)

self.h\_o = nn.Linear(n\_hidden, vocab\_sz)

def forward(self,x):

**h=torch.zeros(64)**

for i in range(3):

h = h + self.i\_h(x\[:,i\])

h = F.relu(self.h\_h(h))

return self.h\_o(h)

| **epoch** | **train\_loss** | **valid\_loss** | **accuracy** | **time** |
| --------- | --------------- | --------------- | ------------ | -------- |
| 0         | 1.727751        | 1.961694        | 0.466366     | 00:02    |
| 1         | 1.392776        | 1.783651        | 0.467792     | 00:02    |
| 2         | 1.41604         | 1.629143        | 0.490135     | 00:02    |
| 3         | 1.376134        | 1.623636        | 0.477775     | 00:02    |

## Search the internet for the GRU architecture and implement it from scratch, and try training a model. See if you can get results similar to those we saw in this chapter. Compare you results to the results of PyTorch's built in GRU module.

![](//images/GRU.png)

Please see my implementation [<span class="underline">here</span>](https://github.com/JayaParmar/Deep-Learning/blob/master/GRU.ipynb)

LSTM gave best accuracy 0.778158 whereas GRU gave an accuracy of 0.811117.

This is because GRU has the reduced number of parameters as compared to LSTM without any compromise whatsoever which results in faster convergence and a more generalized model.

## Take a look at the source code for AWD-LSTM in fastai, and try to map each of the lines of code to the concepts shown in this chapter.

The ‘weight\_p’ parameter is the weight dropout applied as *alpha*

Between two of the inner LSTM, a dropout is applied with probability p as ‘hidden\_p’. This is *beta.*

Alpha and beta are passed as parameters to RNNRegularizer during callback.

AWD LSTM class in fastai can be found [<span class="underline">here</span>](https://docs.fast.ai/text.models.awdlstm).

Activation pre-dropout is TAR and activation post-dropout is AR
