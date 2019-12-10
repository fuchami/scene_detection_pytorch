# coding:utf-8

#%%
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# import logging
# logging.basicConfig(level=logging.INFO)

# load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%% [markdown]
# # トークン化
# ## BERTでは独自のトークナイザーが必要
# - 文頭に[CLS]
# - 文末に[SEP]
#  
#  [CLS] The man went to the store. [SEP] He bought a gallon of milk. [SEP] `

text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)

# %%
list(tokenizer.vocab.keys())[5000:5020]

# Define a new example sentence with multiple meanings of the word "bank"
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the toke strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6}'.format(tup[0], tup[1]))

#%% セグメントID
segments_ids = [1] * len(tokenized_text)
print(segments_ids)

#%% Embedding!
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    encoded_layers = model(tokens_tensor, segments_tensors)

#%%
print("Number of layers:", len(encoded_layers))
layer_i = 0

print("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

# %% こんなことせずとも一発?

text = "Here is the sentence I want embeddings for."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.embeddings.word_embeddings
model.eval()

input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

with torch.no_grad():
    last_hidden_states = model(input_ids)[0]

sentence_embedding = torch.mean(last_hidden_states, dim=0)

# print(sentence_embedding)
print(type(sentence_embedding))
print(sentence_embedding.size())



# %%
