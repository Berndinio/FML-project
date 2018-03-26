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

def generateReviews(path, num):
	n_vocab, int_to_char, char_to_int, chars, total_length = getDataInfo()
	dataX, y = prepareData(n_vocab, char_to_int, 0)
	dataX = dataX * n_vocab
	# define the LSTM model
	model = loadModel(dataX, numpy.reshape(numpy.array(chars), (1,numpy.shape(chars)[0])), path)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# pick a random seed
	allReviews = ""
	for indexFor in range(num):
		start = numpy.random.randint(0, len(dataX)-1)
		pattern = dataX[start]
		pattern = numpy.reshape(pattern, (1, pattern.shape[0]))
		pattern = pattern[0].tolist()
		#print(pattern)
		#print("Seed:")
		#print("\"", ''.join([int_to_char[int(value)] for value in pattern]), "\"")
		#generate characters
		endText = ''.join([int_to_char[int(value)] for value in pattern])
		print("Generating review "+str(indexFor))
		for i in range(500):
			x = numpy.reshape(pattern, (1, len(pattern), 1))
			x = x / float(n_vocab)
			prediction = model.predict(x, verbose=0)
			index = numpy.argmax(prediction)
			result = int_to_char[index]
			seq_in = [int_to_char[int(value)] for value in pattern]
			#print(result, end="")
			endText = endText + result
			#print(result)
			#print("\"", ''.join([int_to_char[int(value)] for value in pattern]), "\"")
			pattern.append(index)
			pattern = pattern[1:len(pattern)]

		allReviews = allReviews + "Review " + str(indexFor) + "\n"
		allReviews = allReviews + endText + "\n \n"
	return allReviews
