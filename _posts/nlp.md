## What is "self-supervised learning"?

Self-supervised learning is training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

## What is a "language model"?

A language model is a model that has been trained to guess what the next word in a text is (having read the ones before).

## Why is a language model considered self-supervised?

A language model is considered self-supervised because we don't need to give labels to our model. Instead it has a process to automatically get labels from the data.

## What are self-supervised models usually used for?

Self-supervised learning is not usually used for the model that is trained directly, but instead is used for pretraining a model used for transfer learning.

## Why do we fine-tune language models?

We fine-tune language models to get even better results. We fine-tune the (sequence-based) language model prior to fine-tuning the classification model.

## What are the three steps to create a state-of-the-art text classifier?

1\. **Tokenization**: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)

2\.**Numericalization**: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab

3.Language model **data loader creation**: fastai provides an LMDataLoader class which automatically handles creating a dependent variable that is offset from the independent variable by one token. It also handles some important details, such as how to shuffle the training data in such a way that the dependent and independent variables maintain their structure as required

4.Language **model creation**: We need a special kind of model that handles input lists which could be arbitrarily big or small.

## How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?

We can use the 50,000 unlabeled movie reviews (along with 50,000 labelled reviews) to fine-tune the pretrained language model, which was trained only on Wikipedia articles. This will result in a language model that is particularly good at predicting the next word of a movie review in the IMDb dataset.

## What are the three steps to prepare your data for a language model?

The first step is to transform the individual texts into a stream by concatenating them together.

Second step is to cut this stream into a certain number of batches (which is our *batch size*).

Third step is to pass this batch size (via data loader) along with TextBlock(DataBlock) which tokenizes and numericalizes text in all batches.

## What is "tokenization"? Why do we need it?

Tokenization converts the text into a list of words. We need to tokenize a text to understand the language intricacies like grammar, abbreviations etc . Every language has a different set of rules which our model should understand while learning from it. Tokenization will help the model learn these rules.

## Name three different approaches to tokenization.

  - > **Word-based:** Split a sentence on spaces, as well as applying language-specific rules to try to separate parts of meaning even when there are no spaces (such as turning "don't" into "do n't"). Generally, punctuation marks are also split into separate tokens.

  - > **Subword based**: Split words into smaller parts, based on the most commonly occurring substrings. For instance, "occasion" might be tokenized as "o c ca sion."

  - > **Character-based**: Split a sentence into its individual characters.

## What is xxbos?

xxbos is a special token that indicates the start of a new text. "BOS" is a standard NLP acronym that means "beginning of stream".

## List four rules that fastai applies to text during tokenization.

Fastai has more than four sets of rules for tokenization. All the rules can be checked using ‘defaults.text\_proc\_rules’. Four of them are mentioned below -

Fix\_html: Replaces special HTML characters with a readable version

replace\_rep: Replaces any character repeated three times or more with a special token for repetition (xxrep), the number of times it's repeated, then the character

replace\_wrep: Replaces any word repeated three times or more with a special token for word repetition (xxwrep), the number of times it's repeated, then the word

spec\_add\_spaces: Adds spaces around / and \#

## Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?

It is done to make it easier for a model to recognize the important parts of a sentence. In a sense, we are translating the original English language sequence into a simplified tokenized language—a language that is designed to be easy for a model to learn.

## What is "numericalization"?

Numericalization is the process of mapping tokens to integers.

## Why might there be words that are replaced with the "unknown word" token?

Words appearing less than 3 times in the most common 60 000 words are replaced with “unknown word” to avoid having an overly large embedding matrix. That can slow down training and use up too much memory, and can also mean that there isn't enough data to train useful representations for rare words.

## With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong\! Be sure to check your answer on the book's website.)

We have cut the ‘stream’ into 64 batches. If the length of the stream was 64 000 tokens, it would divide each batch into a sequence length of 1 000 tokens.

The second row of that tensor (that tensor is first batch) contains the second to 65th token of the first batch. i.e. since it predicts the next word in the sequence it will drop the first token and instead take the 65th token in a column size of 64 tokens.

The second batch contains tokens from 1 001 to 2 000. The first row of this second batch contains the first 64 tokens from this subset i.e. token from 1 001 to 10064.

## Why do we need padding for text classification? Why don't we need it for language modeling?

Text ‘classification’ means to classify text into categories. In our case, category (or sentiment) positive or negative.

Every document in the dataset is a movie review. Each review is different length.To make a PyTorch tensor we need columns of fixed length (or shape) in each row. Hence we need to ‘pad’ smaller documents to make them equal to length of the largest document.

Language modeling unlike text classification predicts the next word in the document. As long as we preserve the sequence of text, we can slice the stream into equal columns (or sequence length). There is no external label to be predicted so the tensor formation needs no padding as the stream is equally divided into rows and columns.

## What does an embedding matrix for NLP contain? What is its shape?

An embedding matrix for NLP contains tokens from vocabulary as rows and index as columns. Max vocab ca be 60000 tokens hence the shape can be max \[60 000, 60 000\]

## What is "perplexity"?

The perplexity metric is often used in NLP for language models: it is the exponential of the loss (i.e., torch.exp(cross\_entropy))

## Why do we have to pass the vocabulary of the language model to the classifier data block?

Language model trains the dataset to predict the next word in the vocabulary. We pass this learnt vocabulary to the model as input to predict the sentiment target of the classifier. This will improve the accuracy of our model.

## What is "gradual unfreezing"?

Gradual unfreezing means unfreezing a few layers at a time to improve the model accuracy. In NLP classifiers, gradual unfreezing is common practice. We do it by code learn. freeze\_to(-n) where n can be -1 or -2 or -3 layers.

## Why is text generation always likely to be ahead of automatic identification of machine-generated texts?

## 

##
