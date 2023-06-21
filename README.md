# Makespeare
Makespeare is a transformer that I coded from scratch and trained on the tiny-shakespeare dataset. This idea is inspired by Andrej Karpathy's video (https://youtu.be/kCc8FmEb1nY) which I used a reference only to overcome certain obstacles. 

The following details the transformer architecture and the things I learnt while implementing it.

## Embedding
In order for the transformer to be able to interpret text, we need to convert the input text into something a computer can understand - numbers. This is done by:

1. Tokenisation: Splitting up text into multiple parts, each part being a **token**
2. Encoding: Giving a unique numerical ID to each unique token. A unique word is thus mapped to a unique numerical ID.
3. Embedding: Converting each token into an n-dimensional vector. This essentially converts words into points in n-dimensional space which can then be played around with. For example, how similar two words are can be measured by the distance between their corresponding points in n-dimensional space (similarity increases the closer the points are).

### Ways to Split Text Up
1. Word-based Tokenisation: Split the input text into its constituent words. (To be continued)
