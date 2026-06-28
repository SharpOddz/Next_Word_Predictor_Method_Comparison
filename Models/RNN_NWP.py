import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import drive
from collections import Counter

#Mount Google Drive
drive.mount('/content/drive')

#Seed setting for reproducibility
SEED = 23
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Load in data from Google Drive
!mkdir -p /content/PTBD
!cp -n /content/drive/MyDrive/PennTreeBankDataset.zip /content/PennTreeBankDataset.zip
!unzip -q -o /content/PennTreeBankDataset.zip -d /content/PTBD

# Updated dataset_dir based on previous output ['ptbdataset']
dataset_dir = "/content/PTBD/ptbdataset"
print("Files in dataset directory:", os.listdir(dataset_dir))

#Tokenize the text
#Every line is a sentence, so the <eos> or "." is added to the end of every sentence
#Whole Word Tokenization (<unk> will represent an unknown word)
def tokenize_text_path(file_path):
  if not os.path.exists(file_path):
    print(f"Warning: File not found at {file_path}")
    return []
  tokens = []
  with open(file_path, 'r', encoding="utf-8") as file:
    for line in file:
      words = line.strip().split()
      if len(words) > 0:
        tokens.extend(words)
        tokens.append('<eos>')
  return tokens

train_tokens = tokenize_text_path(os.path.join(dataset_dir, 'ptb.train.txt'))
valid_tokens = tokenize_text_path(os.path.join(dataset_dir, 'ptb.valid.txt'))
test_tokens = tokenize_text_path(os.path.join(dataset_dir, 'ptb.test.txt'))
#print(train_tokens)


#Build Vocabulary
#Assigning a numerical value to each unique token, <unk> is set uniquely at 0
def build_vocab(tokens, min_freq=1):
  counter = Counter(tokens)
  word_to_id = {
      '<unk>': 0
  }
  for word, count in counter.items():
    if count >= min_freq and word not in word_to_id:
      word_to_id[word] = len(word_to_id)
  id_to_word = {id: word for word, id in word_to_id.items()}
  return word_to_id,id_to_word

word_to_id, id_to_word = build_vocab(train_tokens)
vocab_size = len(word_to_id)

def token_to_id(tokens,word_to_id):
  return [word_to_id.get(token, 0) for token in tokens]

train_ids = token_to_id(train_tokens, word_to_id)
valid_ids = token_to_id(valid_tokens, word_to_id)
test_ids = token_to_id(test_tokens, word_to_id)
#print(train_ids)

#Input-Target Pairs (Fixed Sequence Length)
#Currently implemeted as fixed sequence length but will experiment with dynamic length chunks as well
sequence_length = 25

def generate_input_target_pairs(ids, sequence_length):
  input_sequences = []
  target_sequences = []
  for i in range(0,len(ids)-sequence_length):
    x = ids[i:i+sequence_length]
    y = ids[i+1:i+sequence_length+1]
    input_sequences.append(x)
    target_sequences.append(y)
  return input_sequences, target_sequences

train_x,train_y = np.array(generate_input_target_pairs(train_ids, sequence_length))
val_x,val_y = np.array(generate_input_target_pairs(valid_ids, sequence_length))
test_x,test_y = np.array(generate_input_target_pairs(test_ids, sequence_length))
print(len(train_x))

#Embedding
embedding_dim = 256# vocab_size x embedding dimension matrix

#Printout of pretraining data
print(f"Vocab Size: {vocab_size}")
print(f"Train X Shape: {train_x.shape}")
print(f"Train Y Shape: {train_y.shape}")

#Model Hyperparameters
epochs = 100
batch_size = 32
rnn_units = 256

#RNN Model Architecture
model = keras.Sequential([
    #Embedding layer
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
    #Simple RNN layer
    layers.SimpleRNN(units=rnn_units, return_sequences=True),
    #Dense output layer
    layers.Dense(units=vocab_size, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

#Training RNN
history = model.fit(
    x=train_x,
    y=train_y,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_x, val_y)
)
