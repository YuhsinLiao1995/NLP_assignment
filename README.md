# NLP_assignment

Aspect-Term Polarity Classification in Sentiment Analysis

Names: Léopold Granger, Clara Besnard, Yuhsin Liao
## Type of classification model
We used the RoBerta model (implemented in models.py), proposed in RoBERTa: A Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. Roberta builds on Bert, it is different because it removes the next-sentence pretraining objective and trains with much larger mini-batches and learning rates. 

## Some key information about the model we used
### Implementation of the classifier
The max input size is set to 128 tokens. This is set by the max_len variable, it means that we can process text inputs that are up to 514 tokens long. Embedding size: 768. RoBERTa uses 768-dimensional token embeddings, this refers to the dimensionality of the vector space in which the model represents words and sequences of words as vectors. In other words, each word or sequence of words is represented as a 768-dimensional vector. Number of hidden layers: 12. These are layers of neurons that process the input data and transform it in a way that makes it easier for the model to make predictions. RoBERTa has 12 self-attention layers (Transformer blocks). Vocabulary size: 50,265. RoBERTa has a vocabulary of 50,265 tokens. This is the number of unique words in the model's vocabulary, which it uses to represent the words in its input. The model is pretrained on a large corpus and then fine-tuned on the sentiment analysis dataset. The final layer is replaced with a 3-class classifier for positive, negative and neutral sentiments.

The text sentences are tokenized and encoded using the RobertaTokenizer. This performs the following steps:

1. Sentence splitting - Splits the input text into sentences.
2. Wordpiece tokenization - Breaks up each sentence into wordpieces using Roberta's vocabulary. This results in some words being split up into multiple wordpieces.
3. Adds special tokens - Adds [CLS] to the beginning of each sentence, and [SEP] to the end of each sentence.
4. Positional embeddings - Assigns a position embedding to each wordpiece based on its position in the sentence.
5. Padding - Pads or truncates each sentence to the max_len specified (128 in this case).
6. Encoding - Encodes each wordpiece into an ID from Roberta's vocabulary.

This results in a sequence of IDs, attention masks, and token type IDs which are input to the RoBERTa model.

### Preprocessing
We were originally unsure whether we had to preprocess the sentences before passing them to encoding with Roberta. We experimented with the following preprocessing steps: 
Setting every sentence to lowercase
Removing stopwords
Removing non alphabetical characters
Lemmatizing 

In the end, we only received marginal accuracy gains on the dev set from doing those preprocessing steps. We did see some performance improvements in the time it takes to train the models, possibly because removing some special words (i.e. 1 character words) reduces the amount of work for the model. 

### Hyperparameters tuning
To tune the initial learning rate (in file roberta_imp.ipynb) for a deep learning model, we typically start by testing a range of values to see which works best for our particular problem. In this case, we tested five different learning rates: 1e-5, 2e-5, 3e-5, 4e-5, and 5e-5.

After training the model with each learning rate, we found that the best result was achieved with a learning rate of 2e-5. This value likely worked well for our specific model and problem because it struck a good balance between making small, gradual updates to the model's weights (which can help avoid overshooting the optimal solution) and making updates quickly enough to converge to a good solution in a reasonable amount of time.

Additionally, we implemented an early stopping method within the train function to stop the epoch when the accuracy does not improve 5 times in a row. This technique can help prevent overfitting by stopping the training process when the model starts to memorize the training data rather than learning generalizable patterns and  save time and resources while still achieving good results. Additionally, we fixed the maximum number of epochs to 20. 


### Resources
The classifier uses the transformers library to load and fine-tune the RoBERTa model for sequence classification, and the nltk library is used for stemming and lemmatizing words. The model is trained using the AdamW optimizer and the CrossEntropyLoss loss function, and the learning rate and batch size are configurable using a dictionary config. The RoBERTaDataset class is used to preprocess the dataset, and DataLoader is used to create data loaders. The train method trains the model on the training set and evaluates it on the validation set, while the predict method is not yet implemented.

### Accuracy on the dev dataset
On the dev data set, we obtained on average an accuracy of 87.34% with std 1.15. 



