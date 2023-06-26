# Makespeare
Makespeare is a GPT-style transformer that I coded from scratch and trained on the tiny-shakespeare dataset. This idea is inspired by Andrej Karpathy's video (https://youtu.be/kCc8FmEb1nY) which I used a reference only to overcome certain obstacles. 

## 🛠️ Tools
<img src='https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'> <img src='https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white'> <img src='https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'>

## 📑 Data
The transformer was trained on the `tiny-shakespeare` dataset containing 40,000 lines of text from Shakespeare's plays. Click [here](https://raw.githubusercontent.com/bl0nder/makespeare_datasets/main/shakespeare_input.txt) for the dataset.

An excerpt from the dataset:
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
...
```

## Transformer Architecture
### Embedding
For the transformer to be able to interpret text, we need to convert the input text into something a computer can understand - :sparkles:Numbers:sparkles:. This is done by:

#### 1. Tokenisation
- Splitting up text into multiple parts or **tokens**
#### 2. Encoding:
- Giving a unique numerical ID to each unique token
- Thus, every unique word is mapped to a unique numerical ID.
- In practice, a dictionary is used to keep track of the ID of each word. The number of word-ID pairs present in the dictionary is known as its **vocabulary size** (referred to as `vocab_size` in the code).
  
| Word  | ID |
| ------------- | ------------- |
| Cat  | 1  |
| Dog  | 2  |
| ... | ...|

- If a word that is not present in the dictionary is encountered, special rules are followed to assign an ID to it.
#### 3. Vectorisation: 
- Converting each token into a learnable n-dimensional vector
- For example, how similar two words are can be measured by the distance between their corresponding points in n-dimensional space (similarity increases the closer the points are).
- The dimension of each such vector is fixed and corresponds to `embedding_len` in the code. Some sources also refer to this as `d_model` (model dimension).

