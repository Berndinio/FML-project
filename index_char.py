# extract features with pretrained RNN (transfer learning)
# knn for prediction
# train RNN for prediction
# Naive Bias on character predicition
#
import json
import gzip
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import os.path
import pickle
import os


global uniqueCats
global listNullCats

#######################
# SOME VARIABLES
#######################
v_lastWordsSize = 10
v_sampleReviewSize = 100
v_samplePackageSize = 100
v_feedbackFrequency = 100
v_version = 2
v_splitString = " "
v_replaceInString = ["&quot", "\"", "(", ")", ";"]
v_learningAlgorithm = "" #not implemented yet
v_maxReview = 200
# features
v_activateCategories = False
v_activateRating = True
v_activateName = False # not implemented yet
v_activatePrice = True
#######################
# DATA LOADING
#######################
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def dataToList(path, start, end):
    metaData = []
    for counter, item in enumerate(parse(path)):
        if (counter >= end):
            return metaData
        if (counter >= start):
            metaData.append(item)
    return metaData

def dataToDict(path):
    metaData = {}
    for item in parse(path):
        metaData[item["asin"]] = item
    return metaData

#######################
# EXTRACT FEATURES
#######################

def getUniqueCategories(metaData):
    output = set([])
    for key in metaData.keys():
        item = metaData[key]
        for it1 in item["categories"]:
            for it2 in it1:
                output.add(it2)
    return sorted(list(output))

def getUniqueCategoriesSingle(item):
    output = set([])
    for it1 in item["categories"]:
        for it2 in it1:
            output.add(it2)
    return sorted(list(output))

def getUniqueBrandNames(metaData):
    output = set([])
    i=0
    for item in metaData:
        try:
            output.add(item["brand"])
        except:
            i+=1
    print(str(i)+" without brands.")
    return list(output)


#######################
# SAVING FEATURES
#######################

def saveObject(obj, restFilePath ,version=0):
    if (version == 0):
        print("please enter valid version number")
    filepath = "savings/" + str(version) + "/" + restFilePath
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    f = open(filepath, 'wb+')
    pickle.dump(obj, f)
    f.close()
    print("Done saving!")

def loadObject(restFilePath ,version=0):
    filepath = "savings/" + str(version) + "/" + restFilePath
    if (version == 0) or not os.path.isfile(filepath):
        print("please enter valid version number")
        return None
    f = open(filepath, 'rb')
    objectsLoaded = pickle.load(f)
    return objectsLoaded

def saveClassifier(classifier, filename ,version=0):
    print("saving classifier to file, version = " + str(version))
    saveObject(classifier, "classifier/"+filename, version)
    print("Done saving!")

def saveWorddic(worddic, filename ,version=0):
    print("saving worddic to file, version = " + str(version))
    saveObject(worddic, "worddic/"+filename, version)
    print("Done saving!")

def loadClassifier(filename, version=0):
    print("loading classifier from file, version = " + str(version))
    objectsLoaded = loadObject("classifier/"+filename, version)
    print("Done loading!")
    return objectsLoaded

def loadWorddic(filename, version=0):
    print("loading worddic from file, version = " + str(version))
    objectsLoaded = loadObject("worddic/"+filename, version)
    print("Done loading!")
    return objectsLoaded

#######################
# LEARNING
#######################

def generateClassifier():
    print("Generating classifier")
    return MultinomialNB()

def generateFeatures(featurelist, lastwords, catFeatures, stars, price, name):
    categories_is_active = featurelist[0]
    rating_is_active = featurelist[1]
    title_is_active = featurelist[2]
    price_is_active = featurelist[3]
    sample = []
    if categories_is_active:
        sample.extend(catFeatures)
    if rating_is_active:
        sample.append(stars)
    if title_is_active:
        pass
    if price_is_active:
        price = int(price/10)
        sample.append(price)
    sample.extend(lastwords)
    return sample

def trainClassifier(classifier, samples, labels, worddicSize):
    print("Training classifier...")
    classifier.partial_fit(samples, labels, range(worddicSize))
    print("Done partial training!")

def generateSamples(start, end, metaData, worddic, featurelist):
    global uniqueCats
    global listNullCats

    samples = []
    labels = []
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", start, end)
    for epoch, review in enumerate(reviewsData):
        if (epoch % v_feedbackFrequency == 0):
            print("samples generated:" + str(start+epoch)+"/"+str(v_sampleReviewSize))
        lastwords = []
        for i in range(v_lastWordsSize):
            lastwords.append(2)
        text = review["reviewText"]
        for replacement in v_replaceInString:
            text = text.replace(replacement, "")
        stars = review["overall"]
        product = review["asin"]
        name = metaData[product]["title"]
        price = metaData[product]["price"]
        categories = getUniqueCategoriesSingle(metaData[review["asin"]])

        #generate first sample
        text = '\n'.join(' '.join(line.split()) for line in text.splitlines())
        text = " "+text.lower()
        #text = " asdfg"
        if(text==None):
            continue
        for i,word in enumerate(text):
            index = worddic.index(word)
            lastwords.append(index)
            lastwords.pop(0)

            #generate categories vector
            catFeatures = deepcopy(listNullCats)
            for cat in categories:
                idx = uniqueCats.index(cat)
                catFeatures[idx] = 1.0

            if i==0:
                lastwords.pop()
                lastwords.append(2)
                #labels.append(0)
                next_word = text[i+1]
                index = worddic.index(next_word)
                labels.append(index)
            elif(i==len(text)-1):
                #eot
                #label is #EndOfText
                labels.append(1)
            else:
                next_word = text[i+1]
                index = worddic.index(next_word)
                labels.append(index)
            sample = generateFeatures(featurelist, lastwords, catFeatures, stars, price, name)
            #print(sample, labels[-1])
            samples.append(np.array(sample))
    #print("Done!")
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

#######################
# GENERATE A REV
#######################

def generateReview(classifier, product, featurelist, worddictionary, randomness=False):
    global uniqueCats
    global listNullCats

    #initialize last words
    lastwords = []
    for i in range(v_lastWordsSize):
        lastwords.append(2)

    #extracting product features
    categories = product["categories"]
    stars = product["stars"]
    price = product["price"]
    name = product["name"]

    #generate categories vector
    catFeatures = deepcopy(listNullCats)
    for cat in categories:
        idx = uniqueCats.index(cat)
        catFeatures[idx] = 1.0

    text = [" "]
    #print(worddic)
    while (text[-1]!=worddic[1] and len(text) < v_maxReview):
        sample = generateFeatures(featurelist, lastwords, catFeatures, stars, price, name)
        print(sample)
        if(True):
            prediction = classifier.predict(np.array([np.array(sample)]))
            print(prediction)
            word = worddictionary[prediction[0]]
            text.append(word)
            #print(word, idx, lastwords, prediction)
            lastwords.pop(0)
            lastwords.append(prediction[0])
        else:
            prediction = classifier.predict_proba(np.array([np.array(sample)]))
            idx = np.argsort(prediction)
            word = worddictionary[idx[0][-2]]
            text.append(word)
            #print(word, idx, lastwords, prediction)
            lastwords.pop(0)
            lastwords.append(idx[0][-2])
    return text

#######################
# MAIN LOOP
#######################

if __name__ == '__main__':
    global uniqueCats
    global listNullCats

    # some loading
    print("reading meta data ...")
    metaData = dataToDict("data/music/meta_Digital_Music.json.gz")
    print("Done!")
    print("reading review Data ...")
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", 0, v_sampleReviewSize)
    revLen = len(reviewsData)
    print("Done!")


    # finding categories
    print("unification of categories ...")
    uniqueCats = getUniqueCategories(metaData)
    print("Done!")
    lenUniqueCats = len(uniqueCats)
    listNullCats = []
    for i in range(lenUniqueCats):
        listNullCats.append(0)

    # creating dictionary of words
    print("creating worddic...")
    worddic= ["#beginningOfText", "#endOfText", "#None"]
    for epoch, review in enumerate(reviewsData):
        if (epoch % v_feedbackFrequency == 0):
            print(str(epoch)+"/"+str(revLen))
        text = review["reviewText"].lower()
        for replacement in v_replaceInString:
            text = text.replace(replacement, "")
        for word in text:
            if word not in worddic:
                worddic.append(word)
    print("Done! Created worddic with size " + str(len(worddic)))

    # chosen Classifier
    clf = generateClassifier()
    sampletype = [v_activateCategories, v_activateRating, v_activateName, v_activatePrice]

    # step-wise learning
    for i in range (int(v_sampleReviewSize / v_samplePackageSize)+1):
        start = i * v_samplePackageSize
        end = min((i+1)*v_samplePackageSize, v_sampleReviewSize)
        if (start == end):
            break
        samples, labels = generateSamples(start, end, metaData, worddic, sampletype)
        trainClassifier(clf, np.array(samples), np.array(labels), len(worddic))
        saveClassifier(clf, str(i*v_samplePackageSize)+".nb" , v_version)


    print("generating test review...")
    review_product = {"categories":[uniqueCats[0]], "price":9.99, "stars":3.0, "name":"Ulala"}
    review_text = generateReview(clf, review_product, sampletype, worddic, False)
    text = "".join(review_text)
    print("Done! The review is following:")
    print(text)
    #print("test prediction...")
    #testsamples, testlabel = generateSamples(1, 2, metaData, worddic, sampletype)
    #labelidx = clf.predict(np.array([testsamples[0]]))
    #print("Done!")
    #print(labelidx, worddic[labelidx[0]])

    #clf2 = loadClassifier(str(9*v_samplePackageSize)+".nb", v_version)
    #print("test prediction with loaded classifier...")
    #labelidx = clf2.predict(np.array([testsamples[0]]))
    #print("Done!")
    #print(labelidx, worddic[labelidx[0]])

    #uniqueCategories = getUniqueCategories(metaData)
    #uniqueBrandNames = getUniqueBrandNames(metaData)

    #print(metaData)
    #imageFeatureData = open("data/baby/image_features_Baby.b", "rb")
