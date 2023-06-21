# Makespeare
Makespeare is a transformer that I coded from scratch and trained on the tiny-shakespeare dataset. This idea is inspired by Andrej Karpathy's video (https://youtu.be/kCc8FmEb1nY) which I used a reference only to overcome certain obstacles. 

The following details the transformer architecture and the things I learnt while implementing it.

## Tokenisation
The input data was a text file containing a number of Shakespeare's plays. Passing the entire data on to the model is of little use since that would be the equivalent of training the model on a single sample with over 1 million characters. Instead, if the data is partitioned into several parts (**Tokens**), then sampling these tokens can lead to the creation of a much larger set of samples to train the model on. The process of paritioning the data in the aforementioned manner is known as **Tokenisation**. 

Here are a few (fairly intuitive) ways to tokenise a given piece of text:

1. Word-based Tokenisation: We can simply split the text data into its constituent words (To be continued...)
