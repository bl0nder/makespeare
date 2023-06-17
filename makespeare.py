#Import important libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#Define hyperparameters - These are the hyperparameters that have given me the best result yet
batch_size = 32  #Num of sentences being processed in parallel
context_length = 256  #Num of tokens processed at a time (how much context is there behind understanding each token)
embedding_len = 128 #Each token is converted into an embedding_len dimensional tensor once it undergoes embedding
num_heads = 8 #Num of heads that the embedding matrices will be split in while computing attention
num_encoder_blocks = 1 
num_decoder_blocks = 2  
learning_rate = 5e-5  
max_iterations = 150000 #Num of iterations for which model is trained
eval_interval = 500 #Num of iterations after which validation loss is computed (during model training)
val_iterations = 200 
checkpoint_interval = 10000 #Num of iterations after which a checkpoint is created
num_generated_tokens = 10000  #Num of tokens generated from a trained model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#Download dataset (I have another public repo just for datasets I used for this project)
!wget 'https://raw.githubusercontent.com/bl0nder/makespeare_datasets/main/shakespeare_input.txt'

#Read the dataset
with open('shakespeare_input.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

#------------TOKENISATION------------#
#Character-Level Tokenization
char_list = sorted(list(set(input_text)))
vocab_size = len(char_list)

char_to_token = {}
token_to_char = {}

for i,c in enumerate(char_list):
  char_to_token[c] = i
  token_to_char[i] = c

#Function to encode string into tokens
def encode(string):
  tokens = []
  for c in string:
    tokens.append(char_to_token[c])
  return tokens

#Function to decode tokens into corresponding characters
def decode(tokens):
  chars = []
  for i in tokens:
    chars.append(token_to_char[i])
  return ''.join(chars)

#Convert token array to tensor for further processing
token_ids = torch.tensor(encode(input_text))

#Train/val split
train_idx = int(len(token_ids)*0.9)
train_data = token_ids[0:train_idx]
val_data = token_ids[train_idx:]

#------------MINI-BATCH SELECTION------------#
def minibatch(train_data, val_data, context_length, batch_size, train=True):

  #Selecting whether to sample from training or validation data
  if (train):
    data = train_data
  else:
    data = val_data

  #Random index to pick minibatch from
  ind = torch.randint(0, len(data) - context_length, size = (batch_size,))

  #Create minibatch
  x_batch = torch.stack([data[i : i+context_length] for i in ind])  #Tokens
  y_batch = torch.stack([data[i+1 : i+context_length+1] for i in ind])  #Next tokens in sentence

  x_batch = x_batch.to(DEVICE)
  y_batch = y_batch.to(DEVICE)

  return x_batch, y_batch

#------------EMBEDDING------------#
class InputEmbedding(nn.Module):
  def __init__(self, context_length):
    super(InputEmbedding, self).__init__()
    self.embedding_layer = nn.Embedding(vocab_size, embedding_len).to(DEVICE)
    self.pos_embedding_layer = nn.Embedding(context_length, embedding_len).to(DEVICE)
    self.context_length = context_length

    #Weight initialisation
    torch.nn.init.normal_(self.embedding_layer.weight, mean=0, std=0.02)
    torch.nn.init.normal_(self.pos_embedding_layer.weight, mean=0, std=0.02)

  def forward(self, token_ids, target=False):
    #Token & positional embeddings
    token_embedding = self.embedding_layer(token_ids).to(DEVICE)
    pos_indices = torch.arange(self.context_length).to(DEVICE)
    pos_embedding = self.pos_embedding_layer(pos_indices).to(DEVICE)
    final_embedding = token_embedding + pos_embedding
    return final_embedding

#------------ATTENTION!------------#
class MultiHeadAttention(nn.Module):
  def __init__(self, batch_size, embedding_len, num_heads, dropout_prob=0.2, attention_mask=False):
    super(MultiHeadAttention, self).__init__()

    self.batch_size = batch_size
    self.embedding_len = embedding_len
    self.num_heads = num_heads
    self.head_dim = embedding_len // num_heads
    self.attention_mask = attention_mask

    #Embedding length needs to be divisible by # of heads
    assert (self.head_dim == float(embedding_len/num_heads)), "embedding_len must be divisible by num_heads"

    #Linear layers to compute Wq, Wk, Wv
    self.W_q = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)
    self.W_k = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)
    self.W_v = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)

    #Linear layer + Dropout for output
    self.output = nn.Linear(embedding_len, embedding_len).to(DEVICE)
    self.output_dropout = nn.Dropout(dropout_prob)

    #Weight initialisation for nn layers
    torch.nn.init.normal_(self.W_q.weight, mean=0, std=0.02)
    torch.nn.init.normal_(self.W_k.weight, mean=0, std=0.02)
    torch.nn.init.normal_(self.W_v.weight, mean=0, std=0.02)

  def forward(self, v, k, q):

    #Compute Values, Keys and Queries
    V = self.W_v(v).to(DEVICE)
    K = self.W_k(k).to(DEVICE)
    Q = self.W_q(q).to(DEVICE)

    #Split into num_heads heads for multi-head processing
    V_split, Q_split, K_split = self.split(V,Q,K)

    #Compute scaled dot-product attention
    attention, attention_weights = self.scaled_dot_product_attention(V_split, K_split, Q_split)

    #Concatenate heads
    attention_concat = self.concat_heads(attention)

    #Pass attention through linear layer
    mha_output = self.output_dropout(self.output(attention_concat))
    return mha_output

  def split(self, V, Q, K):

    #Splitting values, keys and queries into num_head heads
    V_split = torch.stack(torch.split(V, self.head_dim, dim=2), dim = 1)
    Q_split = torch.stack(torch.split(Q, self.head_dim, dim=2), dim = 1)
    K_split = torch.stack(torch.split(K, self.head_dim, dim=2), dim = 1)

    return V_split, Q_split, K_split

  def concat_heads(self, attention):

    #This is better understood with a diagram so here it is:
    #[[1 2 3
    #  1 2 3    <- Head #1
    #  1 2 3]
    # [4 5 6
    #  4 5 6    <- Head #2
    #  4 5 6]
    # [7 8 9
    #  7 8 9    <- Head #3
    #  7 8 9]]
    #We wanna transpose the matrix such that we get:
    #[[1 2 3
    #  4 5 6  <- First row of each head
    #  7 8 9]
    # [1 2 3
    #  4 5 6  <- Second row of each head
    #  7 8 9]
    # [1 2 3
    #  4 5 6  <- Third row of each head
    #  7 8 9]]

    attention_concat = attention.transpose(1,2)

    #Now we just wanna 'stretch out' the heads to get the concatenated attention matrix:
    #[[1 2 3 4 5 6 7 8 9] <- First row matrix stretched out
    # [1 2 3 4 5 6 7 8 9] <- Second row matrix stretched out
    # [1 2 3 4 5 6 7 8 9]] <- Third row matrix stretched out

    attention_concat = attention_concat.reshape(batch_size, context_length, -1)
    return attention_concat

  def scaled_dot_product_attention(self, V, K, Q):

    #Attention = Softmax(QK.T/sqrt(d_k))*V

    K_T = torch.transpose(K, -2, -1)
    QK = torch.einsum('abij, abjk -> abik', [Q, K_T])

    #Look-ahead mask
    if (self.attention_mask == True):
      mask = torch.tril(torch.ones((context_length, context_length))).expand(num_heads, context_length, context_length).to(DEVICE)
      mask = mask.expand(self.batch_size, self.num_heads, context_length, context_length)
      QK = QK.masked_fill(mask==0, float('-inf'))

    d_k = K.shape[-1]
    product = QK/np.sqrt(d_k)

    temp = nn.Softmax(dim=-1)
    attention_weights = temp(product)
    attention = torch.einsum("abij, abjk -> abik", [attention_weights, V])

    return attention, attention_weights
  
#------------ENCODER------------# 
class EncoderBlock(nn.Module):
  def __init__(self, vocab_size, context_length, embedding_len, dropout_prob = 0.05):

    super(EncoderBlock, self).__init__()

    #Attributes
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.embedding_len = embedding_len
    self.dropout_prob = dropout_prob
    self.fnn_factor = 4

    #Required NN layers
    self.multi_head_self_attention_layer = MultiHeadAttention(batch_size, self.embedding_len, num_heads).to(DEVICE)
    self.normalisation_mhsa = nn.LayerNorm(embedding_len).to(DEVICE)
    self.normalisation_fnn = nn.LayerNorm((batch_size, self.context_length, self.embedding_len)).to(DEVICE)
    self.fnn = nn.Sequential(
        nn.Linear(embedding_len, embedding_len*self.fnn_factor),
        nn.ReLU(),
        nn.Linear(embedding_len*self.fnn_factor, embedding_len),
        nn.Dropout(self.dropout_prob)
    ).to(DEVICE)
    
    #Weight initialisation
    self.fnn.apply(self._init_weights)

  def _init_weights(self, module):
    if type(module) == nn.Linear:
      torch.nn.init.normal_(module.weight, mean=0, std=0.02)

  def forward(self, x, batch_size, num_heads, verbose=False):

    #Add & Pre-Norm
    mhsa_pre_norm = self.normalisation_mhsa(x)  #Even though the original paper uses normalisation after computing self-attention, pre-normalisation may produce better results (and it did in this case) 
    mhsa = self.multi_head_self_attention_layer(mhsa_pre_norm, mhsa_pre_norm, mhsa_pre_norm)
    mhsa_output = (mhsa + x)

    #Feed-forward NN
    fnn_pre_norm = self.normalisation_fnn(mhsa_output)
    fnn = self.fnn(fnn_pre_norm)
    fnn_output = fnn + mhsa_output

    return fnn_output

#Encoder class
class Encoder(nn.Module):
  def __init__(self, num_encoder_blocks, batch_size, vocab_size, context_length, embedding_len, num_heads, dropout_prob = 0.05):

    super(Encoder, self).__init__()

    #Attributes
    self.num_encoder_blocks = num_encoder_blocks
    self.batch_size = batch_size
    self.num_heads = num_heads
    self.encoder_blocks = []
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.embedding_len = embedding_len
    self.dropout_prob = dropout_prob

    #List of encoder blocks
    self.encoder_blocks = nn.ModuleList([EncoderBlock(vocab_size, context_length, embedding_len, self.dropout_prob) for i in range(self.num_encoder_blocks)])

  def forward(self, x):
    encoder_input = x
    for i, block in enumerate(self.encoder_blocks):
      encoder_input = block(encoder_input, self.batch_size, self.num_heads).to(DEVICE)


    encoder_output = encoder_input
    return encoder_output

#------------DECODER------------#
class DecoderBlock(nn.Module):
  def __init__(self, batch_size, vocab_size, context_length, embedding_len, num_heads, dropout_prob = 0.05):
    super(DecoderBlock, self).__init__()
    
    #Attributes
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.embedding_len = embedding_len
    self.num_heads = num_heads
    self.dropout_prob = dropout_prob
    self.fnn_factor = 4
    
    #NN Layers
    #Masked MHSA
    self.masked_mhsa_layer = MultiHeadAttention(batch_size, embedding_len, num_heads, 0.2, True)
    self.normalisation_mhsa = nn.LayerNorm(embedding_len)

    #Cross attention (Uncomment in encoder-decoder transformer)
    # self.cross_mha_layer = MultiHeadAttention(batch_size, embedding_len, num_heads, 0.2)
    # self.normalisation_cross_mha = nn.LayerNorm(embedding_len)
    
    #Feed-forward NN
    self.normalisation_fnn = nn.LayerNorm(embedding_len)
    self.fnn = nn.Sequential(
        nn.Linear(embedding_len, embedding_len*self.fnn_factor),
        nn.ReLU(),
        nn.Linear(embedding_len*self.fnn_factor, embedding_len),
        nn.Dropout(self.dropout_prob)
    )

    #Weight initialisation
    self.fnn.apply(self._init_weights)

  def _init_weights(self, module):
    if type(module) == nn.Linear:
      torch.nn.init.normal_(module.weight, mean=0, std=0.02)

  def forward(self, x, q_cross, k_cross):
    #Masked multi-head self attention
    masked_mhsa_pre_norm = self.normalisation_mhsa(x)
    masked_mhsa = self.masked_mhsa_layer(masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm)
    masked_mhsa_output = masked_mhsa + x

    #Multi-head cross attention - Uncomment the following 3 lines if using encoder-decoder transformer. Redundant in decoder-only model (such as this one) since there is no encoder output to calculate cross attention with
    # cross_mha_pre_norm = self.normalisation_cross_mha(masked_mhsa_output)
    # cross_mha = self.cross_mha_layer(cross_mha_pre_norm, k_cross, q_cross)
    # cross_mha_output = cross_mha + masked_mhsa_output
      
    #Feedforward NN
    fnn_pre_norm = self.normalisation_fnn(masked_mhsa_output)    #If cross attention is being used, replace masked_mhsa_output here with cross_mha_output
    fnn = self.fnn(fnn_pre_norm)
    fnn_output = fnn + masked_mhsa_output

    return fnn_output
  
class Decoder(nn.Module):
  def __init__(self, num_decoder_blocks, batch_size, vocab_size, context_length, embedding_len, num_heads, dropout_prob = 0.05):
    super(Decoder, self).__init__()

    #Attributes
    self.num_decoder_blocks = num_decoder_blocks
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.embedding_len = embedding_len
    self.num_heads = num_heads
    self.dropout_prob = dropout_prob
    
    #List of decoder blocks
    self.decoder_blocks = nn.ModuleList([DecoderBlock(batch_size, vocab_size, context_length, embedding_len, num_heads, dropout_prob) for i in range(num_decoder_blocks)])

  def forward(self, x):
    #Loop through all decoder blocks and process inputs sequentially (output of a block is input to the next)
    decoder_input = x
    for i, block in enumerate(self.decoder_blocks):
      decoder_input = block(decoder_input, x_embedding, x_embedding)

    decoder_output = decoder_input
    return decoder_output

#------------TRANSFORMER------------#
class Transformer(nn.Module):
    def __init__(self):
      super(Transformer, self).__init__()
      self.decoder = Decoder(num_decoder_blocks, batch_size, vocab_size, context_length, embedding_len, num_heads, 0.2)
      
      #NN Layers
      self.normalisation = nn.LayerNorm(embedding_len) # final layer norm
      self.linear = nn.Linear(embedding_len, vocab_size)
      
      #Token embedding
      self.input_embedding = InputEmbedding(context_length)

      #Weight Initialisation
      self.apply(self._init_weights)

    def _init_weights(self, module):
      if type(module), == nn.Linear:
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      elif type(module) == nn.Embedding:
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
      #Input embeddings
      x_embeddings = self.input_embedding(x)  

      #Uncomment the following line if using an encoder-decoder model
      # encoder_output = self.encoder(x_embeddings)
      decoder_output = self.decoder(x_embeddings)  #Replace x_embeddings with encoder_output if using an encoder-decoder model
      normalised_decoder_output = self.normalisation(decoder_output)
      logits = self.linear(normalised_decoder_output)
    
      #If targets are given, compute loss
      if targets is None:
        loss = None
      else:
        logits = logits.reshape(batch_size, context_length, -1)
        targets = targets.reshape(batch_size*context_length, -1)
        loss = F.cross_entropy(logits, targets)

      return logits, loss

    def generate(self, x, max_new_tokens):
      for i in range(max_new_tokens):
        x_latest = x[:, -context_length:]
        logits, loss = self(x_latest)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        x_next = torch.multinomial(probs, num_samples=1).reshape(batch_size, 1)
        print(token_to_char[x_next[-1].item()], end='')
        x = torch.cat((x, x_next), dim=1)
      return x

#------------TRAINING------------#
#Function to compute validation loss
@torch.no_grad()
def val_loss (model, val_iterations):
  with torch.no_grad():
    out = {
        'train' : 0,
        'val' : 0
    }
    model.eval()

    for i in range(2):
      for j in range(val_iterations):
        if (i == 0):
          x,y = minibatch(train_data, val_data, context_length, batch_size)
        else:
          x,y = minibatch(train_data, val_data, context_length, batch_size, train=False)

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits, cross_entropy_loss = model(x,y)

        if (i==0):
          out['train'] += cross_entropy_loss
        else:
          out['val'] += cross_entropy_loss

    out['train'] /= val_iterations
    out['val'] /= val_iterations

    model.train()
    return out

transformer = Transformer().to(DEVICE)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
print(sum(p.numel() for p in transformer.parameters())/1e6, 'M parameters') #Number of params in model

#Training loop
for i in range(max_iterations):
  
  #After every eval_interval iterations, compute validation loss
  if (i+1) % eval_interval == 0:
    losses = val_loss(transformer, val_iterations)
    print(f"step {i+1}: train loss {losses['train']}, val loss {losses['val']}")

  #Every checkpoint_interval iterations, create a checkpoint for the model, i.e, save the model state dictionary (along with other info if you want) somewhere
  if ((i+1) % checkpoint_interval == 0):
    checkpoint = {
    'iterations': i+1,
    'num_encoder_blocks': num_encoder_blocks,
    'num_decoder_blocks': num_decoder_blocks,
    'state_dict': transformer.state_dict()  #Most important thing to save
    }
    torch.save(checkpoint, f'models/checkpoint_ctx{context_length}_iter{i+1}_character_encoding.pth')

  #Get minibatch of training data and compute loss
  x, y = minibatch(train_data, val_data, context_length, batch_size, True)
  logits, loss = transformer(x, y)

  #Learn
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

#------------TEXT GENERATION------------#
#Using a pre-trained model by loading a checkpoint
model = Transformer().to(DEVICE)
state_dict = torch.load('models/checkpoint_ctx256_iter150000_character_encoding.pth') #Load saved model  
model.load_state_dict(state_dict['state_dict']) #Load state dictionary into model

#Generating Shakespearean text
context = torch.ones((batch_size,context_length), dtype=torch.long, device=DEVICE)
context *= 8  #Token for full-stop
gen_output = decode(model.generate(context, max_new_tokens = num_generated_tokens)[0].tolist())
