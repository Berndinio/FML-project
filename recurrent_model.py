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

# main control variables
v_sampleReviewSize = 836006
v_samplePackageSize = 1000
v_feedbackFrequency = 10000
v_sequenceLength = 100

# text manipulation variables
v_messageStart = chr(2)
v_messageEnd = chr(3)
v_chooseTrainingRatingRangeStart = 1.0
v_chooseTrainingRatingRangeEnd = 2.0
v_chooseTrainingHelpfulRangeStart = 0.0
v_chooseTrainingHelpfulRangeEnd = 1.0
v_minTrainingHelpful = 3.0
v_chooseTrainingWordsRangeStart = 20
v_chooseTrainingWordsRangeEnd = 320
v_replaceInString = ["&quot;", "\""]
v_manipulateTrainingReplace = True
v_manipulateTrainingLower = False
v_manipulateTrainingRemoveNonASCII = True
v_manipulateTrainingRemoveAdditionalWhitespaces = True

# RNN variables
v_layerSize = 1024
v_activateSecondLayer = True


##############################
# DATA FUNCTIONS
##############################

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def dataToList(path, start, end):
    data = []
    for i, item in enumerate(parse(path)):
        if (i >= end):
            return data
        if (i >= start):
            if (item["overall"] >= v_chooseTrainingRatingRangeStart) and (item["overall"] <= v_chooseTrainingRatingRangeEnd):
                if (len(item["reviewText"]) >= v_chooseTrainingWordsRangeStart) and (len(item["reviewText"]) <= v_chooseTrainingWordsRangeEnd):
                    if(item["helpful"][1] == 0):
                        frac = 0
                    else:
                        frac = item["helpful"][0]/item["helpful"][1]
                    if(v_minTrainingHelpful <= item["helpful"][1] and v_chooseTrainingHelpfulRangeStart <= frac and v_chooseTrainingHelpfulRangeEnd >= frac):
                        data.append(item)
    return data

def getDataInfo():
    # load ascii text and covert to lowercase
    print("reading review Data ...")
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", 0, 836006)
    raw_text = ""
    chars = []
    total_length = 0
    for epoch, review in enumerate(reviewsData):
        if (epoch % v_feedbackFrequency == 0):
            print("Reviews read: " + str(epoch))
        review_text = review["reviewText"]
        if v_manipulateTrainingRemoveNonASCII:
            review_text = ''.join([x for x in review_text if ord(x) < 128])
        if v_manipulateTrainingReplace:
            for replacement in v_replaceInString:
                review_text = review_text.replace(replacement, "")
        if v_manipulateTrainingRemoveAdditionalWhitespaces:
            review_text = '\n'.join(' '.join(line.split()) for line in review_text.splitlines())
        raw_text = v_messageStart + review_text + v_messageEnd
        if v_manipulateTrainingLower:
            raw_text = raw_text.lower()
        chars = list(set(list(set(raw_text)) + chars))
        total_length = total_length + len(raw_text)
    print("Done!")
    # create mapping of unique chars to integers
    chars = sorted(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data
    n_vocab = len(chars)
    print ("Total Characters ALL: "+ str(n_vocab))
    print(chars)
    return n_vocab, int_to_char, char_to_int, chars, total_length



def prepareData(n_vocab, char_to_int, cycle):
    # load ascii text and covert to lowercase
    print("reading review Data ...")
    start = cycle * v_samplePackageSize
    end = min((cycle+1)*v_samplePackageSize, v_sampleReviewSize)
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", start, end)
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
    # summarize the loaded data
    n_chars = len(raw_text)
    print ("Total Characters: "+ str(n_chars))
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
    return X, y

def generate_batch_by_batch_data(n_vocab, char_to_int, batch_size):
    # load ascii text and covert to lowercase
    #print("reading review Data ...")
    reading_point = 0
    seq_length = v_sequenceLength
    text = ""
    while(reading_point < v_sampleReviewSize):
        end = min(reading_point + v_samplePackageSize, v_sampleReviewSize)
        #print("\n Reading reviewData start: "+str(reading_point)+" end: "+str(end))
        reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", reading_point, end)
        for epoch in range(reading_point, end):
            if (epoch % v_feedbackFrequency == 0):
                print(" - reviews read: " + str(epoch))
        reading_point = end
        raw_text = ""
        for review in reviewsData:
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
        text = text + raw_text
        #print("adding text: " + raw_text)
        while (len(text) > batch_size + seq_length):
            #print("total text size:" + str(len(text)))
            dataX = []
            dataY = []
            for i in range(0, batch_size, 1):
                seq_in = text[i:i + seq_length]
                seq_out = text[i + seq_length]
                dataX.append([char_to_int[char] for char in seq_in])
                dataY.append(char_to_int[seq_out])
            n_patterns = len(dataX)
            # reshape X to be [samples, time steps, features]
            X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
            # normalize
            X = X / float(n_vocab)
            # one hot encode the output variable
            y = np_utils.to_categorical(dataY, num_classes=n_vocab)
            text = text[batch_size:]
            #print("returning batch X with size "+str(numpy.shape(X)))
            #print("returning batch y with size "+str(numpy.shape(y)))
            yield X, y

##############################
# MODEL FUNCTIONS
##############################
def getModel(X, y):
    model = Sequential()
    model.add(LSTM(v_layerSize, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    if v_activateSecondLayer:
        model.add(LSTM(v_layerSize))
        model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model

def loadModel(X, y, weights_path):
    model = getModel(X, y)
    model.load_weights(weights_path)
    print("Model loaded...")
    return model
