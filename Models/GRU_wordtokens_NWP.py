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
      '<pad>': 0,
      '<unk>': 1
  }
  for word, count in counter.items():
    if count >= min_freq and word not in word_to_id:
      word_to_id[word] = len(word_to_id)
  id_to_word = {id: word for word, id in word_to_id.items()}
  return word_to_id,id_to_word

word_to_id, id_to_word = build_vocab(train_tokens)
vocab_size = len(word_to_id)

PAD_ID = word_to_id['<pad>']
UNK_ID = word_to_id['<unk>']
def token_to_id(tokens, word_to_id):
    return [word_to_id.get(token, UNK_ID) for token in tokens]

train_ids = token_to_id(train_tokens, word_to_id)
valid_ids = token_to_id(valid_tokens, word_to_id)
test_ids = token_to_id(test_tokens, word_to_id)
#print(train_ids)

#Input-Target Pairs (Fixed Sequence Length)
#Currently implemeted as fixed sequence length but will experiment with dynamic length chunks as well
sequence_length = 25

def generate_input_target_pairs(ids, sequence_length, stride=1):
    input_sequences = []
    targets = []
    for i in range(0, len(ids) - sequence_length, stride):
        x = ids[i:i + sequence_length]
        y = ids[i + sequence_length]
        input_sequences.append(x)
        targets.append(y)
    return input_sequences, targets

train_x, train_y = generate_input_target_pairs(train_ids, sequence_length)
val_x, val_y = generate_input_target_pairs(valid_ids, sequence_length)
test_x, test_y = generate_input_target_pairs(test_ids, sequence_length)

train_x = np.array(train_x, dtype=np.int32)
train_y = np.array(train_y, dtype=np.int32)

val_x = np.array(val_x, dtype=np.int32)
val_y = np.array(val_y, dtype=np.int32)

test_x = np.array(test_x, dtype=np.int32)
test_y = np.array(test_y, dtype=np.int32)

#Embedding
embedding_dim = 256# vocab_size x embedding dimension matrix

#Printout of pretraining data
print(f"Vocab Size: {vocab_size}")
print(f"Train X Shape: {train_x.shape}")
print(f"Train Y Shape: {train_y.shape}")

from tensorflow.keras import mixed_precision
#Model Hyperparameters
epochs = 100
batch_size = 512
rnn_units = 256

#Mixed precision for GPU speed up
mixed_precision.set_global_policy("mixed_float16")

#RNN Model Architecture
model = keras.Sequential([
    #Embedding layer
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    #Simple GRU layer
    layers.GRU(
        units=rnn_units,
        return_sequences=False,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True
    ),
    #Dense output layer
    layers.Dense(units=vocab_size, dtype="float32")
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'],
              steps_per_execution=25)

model.summary()

#Training RNN
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    ,
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )
]

#Speed Optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .cache()
    .shuffle(10000)
    .batch(batch_size, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_x, val_y))
    .cache()
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

#Train model
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

import matplotlib.pyplot as plt
import pandas as pd

# Convert history to a DataFrame and plot, assigning loss metrics to a secondary y-axis
pd.DataFrame(history.history).plot(
    figsize=(10, 6),
    secondary_y=['loss', 'val_loss'],
    title='Model Metrics over Epochs',
    grid=True
)

plt.xlabel('Epoch')
plt.show()

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_x, test_y))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_loss, test_acc = model.evaluate(test_ds)
test_perplexity = np.exp(test_loss)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Perplexity: {test_perplexity:.2f}")

# Save the model to Google Drive
model_save_path = '/content/drive/MyDrive/ptb_gru_model.keras'
model.save(model_save_path)
print(f"Model successfully saved to {model_save_path}")

import os
from tensorflow import keras
import tensorflow as tf
import numpy as np

"""
  NEXT WORD PREDICITION
Can load in the model from Google Drive but you will still have to run the vocab part at the beginning
"""

#Check if model is already loaded, otherwise load from Drive
try:
    model
except NameError:
    model_save_path = '/content/drive/MyDrive/ptb_gru_model.keras'
    if os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}...")
        model = keras.models.load_model(model_save_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_save_path}. Please train and save the model first.")

PAD_ID = word_to_id['<pad>']
UNK_ID = word_to_id['<unk>']

def predict_next_word(model, text, word_to_id, id_to_word, sequence_length, top_k=5):
    tokens = text.lower().strip().split()
    ids = [word_to_id.get(token, UNK_ID) for token in tokens]

    if len(ids) < sequence_length:
        # cuDNN requires right-padding when using mask_zero=True
        ids = ids + [PAD_ID] * (sequence_length - len(ids))
    else:
        ids = ids[-sequence_length:]

    x = np.array([ids], dtype=np.int32)
    predictions = model.predict(x, verbose=0)
    next_word_logits = predictions[0]
    probs = tf.nn.softmax(next_word_logits).numpy()
    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append((id_to_word[idx], probs[idx]))

    return results

top_predictions = predict_next_word(
    model,
    "the final hidden state is a mix of old memory and new memory",
    word_to_id,
    id_to_word,
    sequence_length
)

for word, prob in top_predictions:
    print(f"{word}: {prob:.4f}")
