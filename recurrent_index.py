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
from keras.callbacks import LambdaCallback

from keras.utils import np_utils

import tensorflow as tf

from recurrent_model import getModel
from recurrent_model import getDataInfo
from recurrent_model import parse
from recurrent_model import getDataInfo
from recurrent_model import generate_batch_by_batch_data
from recurrent_generate import generateReviews
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))





#########################
#count all text lengths
#########################
#max = v_textlength=81575401
v_textlength = 0
n_vocab, int_to_char, char_to_int, chars, total_length = getDataInfo()
v_textlength = total_length
print("Software v_textlength="+str(v_textlength))
#######################
# SOME VARIABLES
#######################
v_batchSize = 256
v_epochs = 0
v_batchesPerEpoch = min(5000, int(v_textlength/v_batchSize))

#######################
# TRAINING LOOP
#######################
print("generate_batch_by_batch_data first call")
init_generator = generate_batch_by_batch_data(n_vocab, char_to_int, v_batchSize)
print(init_generator)
X, y = next(init_generator)

# define the LSTM model
print("Getting the model")
model = getModel(X, y)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="savings-LSTM2/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
print("getting checkpoint")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = LambdaCallback(on_batch_begin=lambda batch,logs: time.sleep(0.1))
callbacks_list = [checkpoint, checkpoint2]
print("generate_batch_by_batch_data second call")

# fit the model
for i in range(v_epochs):
    generator = generate_batch_by_batch_data(n_vocab, char_to_int, v_batchSize)
    print("Fitting model in epoch: "+str(i+1))
    if (i > 0):
        model.fit_generator(generator, steps_per_epoch=v_batchesPerEpoch, epochs=i+1, callbacks=callbacks_list, initial_epoch = i)
    else:
        model.fit_generator(generator, steps_per_epoch=v_batchesPerEpoch, epochs=1, callbacks=callbacks_list)
    #time.sleep(180)

for root, dirs, files in os.walk("savings-LSTM2/"):
    for filename in files:
        print("Generating reviews for '"+filename+"'")
        reviews = generateReviews("savings-LSTM2/"+filename, 10)
        f = open("/home/berndinio/Dropbox/FuML-own/Data/generatedReviews-LSTM2/"+filename, 'w+')
        f.write(reviews)
        f.close()
