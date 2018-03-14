#source: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
#paper: https://arxiv.org/pdf/1708.08151.pdf

# Larger LSTM Network to Generate Text for Amazon reviews
import numpy
import json
import gzip
import os.path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import tensorflow as tf

#######################
# SOME VARIABLES
#######################
# main control variables
v_sampleReviewSize = 10
v_feedbackFrequency = 100
v_sequenceLength = 100

# text manipulation variables
v_messageStart = chr(2)
v_messageEnd = chr(3)
v_chooseTrainingRatingRangeStart = 4.0
v_chooseTrainingRatingRangeEnd = 5.0
v_replaceInString = ["&quot;", "\""]
v_manipulateTrainingReplace = True
v_manipulateTrainingLower = False
v_manipulateTrainingRemoveNonASCII = True
v_manipulateTrainingRemoveAdditionalWhitespaces = True

# RNN variables
v_batchSize = 256
v_layerSize = 1024
v_epochs = 20
v_activateSecondLayer = True

#######################
# SOME FUNCTIONS
#######################

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def dataToList(path, start, end):
    data = []
    counter = 0
    for item in parse(path):
        if (counter >= end):
            return data
        if (counter >= start) and (item["overall"] >= v_chooseTrainingRatingRangeStart) and (item["overall"] <= v_chooseTrainingRatingRangeEnd):
            data.append(item)
            counter +=1
    return data

#######################
# INPUT PREPARATION
#######################

# load ascii text and covert to lowercase
print("reading review Data ...")
reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", 0, v_sampleReviewSize)

raw_text = ""
for epoch, review in enumerate(reviewsData):
    if (epoch % v_feedbackFrequency == 0):
        print(str(epoch)+"/"+str(v_sampleReviewSize))
    review_text = review["reviewText"]
    if v_manipulateTrainingRemoveNonASCII:
        review_text = ''.join([x for x in review_text if ord(x) < 128])
    if v_manipulateTrainingReplace:
        for replacement in v_replaceInString:
            review_text = review_text.replace(replacement, "")
    if v_manipulateTrainingRemoveAdditionalWhitespaces:
        review_text = '\n'.join(' '.join(line.split()) for line in review_text.splitlines())
    raw_text = raw_text + v_messageStart + review_text + v_messageEnd

if v_manipulateTrainingLower:
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

# prepare the dataset of input to output pairs encoded as integers
seq_length = v_sequenceLength
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
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1/10)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
model = Sequential()
model.add(LSTM(v_layerSize, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
if v_activateSecondLayer:
    model.add(LSTM(v_layerSize))
    model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
print("fitting model")
model.fit(X, y, epochs=v_epochs, batch_size=v_batchSize, callbacks=callbacks_list)
