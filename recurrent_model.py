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

def getModel(layerSize, X, y, activateSecondLayer):
    model = Sequential()
    model.add(LSTM(layerSize, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    if activateSecondLayer:
        model.add(LSTM(layerSize))
        model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model
