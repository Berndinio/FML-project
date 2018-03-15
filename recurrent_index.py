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
from recurrent_model import prepareData
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#######################
# SOME VARIABLES
#######################
v_batchSize = 256
v_epochs = 20

#######################
# TRAINING LOOP
#######################
X, y, *rest = prepareData()

# define the LSTM model
model = getModel(X, y)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint

# fit the model
for i in range(v_epochs):
    filepath="savings-LSTM/weights-improvement-"+str(i)+"-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = LambdaCallback(on_batch_begin=lambda batch,logs: time.sleep(0.5))
    callbacks_list = [checkpoint, checkpoint2]
    print("Fitting model in epoch: "+str(i))
    model.fit(X, y, epochs=1, batch_size=v_batchSize, callbacks=callbacks_list)
    #time.sleep(180)
