
# Edit the file directory according to your usage

roberta_config = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/Pytorch roBERTa/config.json'
roberta_model = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/Pytorch roBERTa/pytorch_model.bin'
vocab_file = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/Pytorch roBERTa/vocab.json'
merges_file = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/Pytorch roBERTa/merges.txt'
training_file = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/train_5_fold.csv'
character_dictionary = 'gdrive/My Drive/Competition Data/Tweet Sentiment Extraction/character.json'

#LSTM configs
lstm_hidden_size = 64
lstm_num_layers = 3
lstm_direction = 2 #bidirectional LSTM


#Epochs
token_num_epochs = 5
char_num_epochs = 65


#Learning rates
token_lr = 3e-5
character_lr = 0.003

#batch_size
batch_size = 32