#source: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# Larger LSTM Network to Generate Text for Amazon reviews
import numpy
import json
import gzip
import os.path

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils

v_sampleReviewSize = 100

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def dataToList(path, start, end):
    data = []
    for counter, item in enumerate(parse(path)):
        if (counter >= end):
            return data
        if (counter >= start):
            data.append(item)
    return data

# load ascii text and covert to lowercase
print("reading review Data ...")
reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", 0, v_sampleReviewSize)
message_start = chr(2)
message_end = chr(3)
raw_text = ""
for review in reviewsData:
    raw_text = raw_text + message_start + review["text"] + message_end
raw_text = raw_text.lower()
print("Done!")
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: "+ str(n_chars))
print ("Total Vocab: " + str(n_vocab))

asdasd

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: " + str(n_patterns))
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
