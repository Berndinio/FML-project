# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from recurrent_model import loadModel
from recurrent_model import prepareData
from recurrent_model import getDataInfo


n_vocab, int_to_char, char_to_int, chars, total_length = getDataInfo()

dataX, y = prepareData(n_vocab, char_to_int, 0)
dataX = dataX * n_vocab
# define the LSTM model
print(numpy.shape(dataX))
print(numpy.shape(y))

model = loadModel(dataX, numpy.reshape(numpy.array(chars), (1,96)), "savings-LSTM/weights-improvement-01-3.1903-bigger.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
pattern = numpy.reshape(pattern, (1, pattern.shape[0]))
pattern = pattern[0].tolist()
print(pattern)
print("Seed:")
print("\"", ''.join([int_to_char[int(value)] for value in pattern]), "\"")
# generate characters
for i in range(100):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	print(numpy.shape(x))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[int(value)] for value in pattern]
	#print(result, end="")
	print(result)
	print("\"", ''.join([int_to_char[int(value)] for value in pattern]), "\"")
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
