# Next Word Predictor Method Comparison 
Goal of this project is to compare next word prediction performance across the following deep learning methods: Recurrent Neural Networks, LSTMS, and Transformers on the Penn Tree Bank Dataset.


### Recurrent Neural Network
Will compare different types of RNNs: word-level tokenization vs subword-level tokenization vs character-level tokenization.

The embedding dimension is 256, stride=1 for all methods types compared

### Recurrent Neural Network (RNN) Results 

#### RNN Word-level Tokenizer Results*
*This model run used stride=5, will need to re-run with stride=1 to stay in line with the other methods compared

| Metric | Score |
| :---: | :---: | 
| Test Loss | 5.5466 |
| Test Accuracy | 0.1863 |
| Test Perplexity | 256.37 |

Example Predictions (These sentences don't have a defined next word, not part of the test set): 
| Input Sequence | Model Predicted Next Word |
| :---: | :---: | 
| "the only thing i would definitely add is saving the vocabulary along with" | "a" | 
| "goal of this project is to compare next word prediction performance across the following deep learning" | "the" | 
| "the final hidden state is a mix of old memory and new memory" | <eos> (end of sentence) | 

<img width="875" height="528" alt="image" src="https://github.com/user-attachments/assets/7de63e34-b590-459e-92a0-b751156d534b" />



### Gated Recurrent Units (GRU) Results 

#### GRU Word-level Tokenizer Results

| Metric | Score |
| :---: | :---: | 
| Test Loss | 4.8512 |
| Test Accuracy | 0.2297 |
| Test Perplexity | 127.90 |

Example Predictions (These sentences don't have a defined next word, not part of the test set): 
| Input Sequence | Model Predicted Next Word |
| :---: | :---: | 
| "the only thing i would definitely add is saving the vocabulary along with" | "the" | 
| "goal of this project is to compare next word prediction performance across the following deep learning" | <eos> (end of sentence) | 
| "the final hidden state is a mix of old memory and new memory" | <eos> (end of sentence) | 

<img width="867" height="528" alt="image" src="https://github.com/user-attachments/assets/0ff66830-13bc-4070-90c3-df3f76c07ae6" />

### Long Short Term Memory Model (LSTM) Results 

#### LSTM Word-level Tokenizer Results
This model utilizes dropout as a regularization technique

| Metric | Score |
| :---: | :---: | 
| Test Loss | 4.8680 |
| Test Accuracy | 0.2324 |
| Test Perplexity | 130.06 |

Example Predictions (These sentences don't have a defined next word, not part of the test set): 
| Input Sequence | Model Predicted Next Word |
| :---: | :---: | 
| "the only thing i would definitely add is saving the vocabulary along with" | "the" | 
| "goal of this project is to compare next word prediction performance across the following deep learning" | <eos> (end of sentence) | 
| "the final hidden state is a mix of old memory and new memory" | <eos> (end of sentence) | 

<img width="875" height="528" alt="image" src="https://github.com/user-attachments/assets/08cb735e-7359-4b99-b61a-d4d8e206a5ad" />


