import json
import gzip
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import os.path
import pickle
#######################
# SOME VARIABLES
#######################
v_lastWordsSize = 5
v_sampleReviewSize = 2000
v_samplePackageSize = 200
v_feedbackFrequency = 100
v_version = 1
v_splitString = " "
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

def generateClassifier():
    print("Generating classifier")
    return MultinomialNB()

def trainClassifier(classifier, samples, labels, worddicSize):
    print("Training classifier...")
    classifier.partial_fit(samples, labels, range(worddicSize))
    print("Done partial training!")

def saveClassifier(classifier, version=0):
    print("saving classifier to file, verion = " + str(version))
    if (version == 0):
        print("please enter valid version number")
    filepath = "savings/" + str(version) + "/classifier"
    f = open(filepath, 'wb+')
    pickle.dump(classifier, f)
    f.close()
    print("Done saving!")

def loadClassifier(version=0):
    print("loading classifier from file, verion = " + str(version))
    filepath = "savings/" + str(version) + "/classifier"
    if (version == 0) or not os.path.isfile(filepath):
        print("please enter valid version number")
        return None
    f = open(filepath, 'rb')
    objectsLoaded = pickle.load(f)
    return objectsLoaded
    print("Done loading!")

def saveWorddic(worddic, version=0):
    print("saving worddic to file, verion = " + str(version))
    if (version == 0):
        print("please enter valid version number")
    filepath = "savings/" + str(version) + "/worddic"
    f = open(filepath, 'wb+')
    pickle.dump(worddic, f)
    f.close()
    print("Done saving!")

def loadWorddic(version=0):
    print("loading worddic from file, verion = " + str(version))
    filepath = "savings/" + str(version) + "/worddic"
    if (version == 0) or not os.path.isfile(filepath):
        print("please enter valid version number")
        return None
    f = open(filepath, 'rb')
    objectsLoaded = pickle.load(f)
    return objectsLoaded
    print("Done loading!")

def generateSamples(start, end, reviewsData, metaData, worddic):
    samples = []
    labels = []
    #print("loading new reviews...")
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", start, end)
    #print("start generating samples ...")
    package_size = end - start
    for epoch, review in enumerate(reviewsData):
        if (epoch % v_feedbackFrequency == 0):
            print("samples generated:" + str(start+epoch)+"/"+str(v_sampleReviewSize))
        lastwords = []
        for i in range(v_lastWordsSize):
            lastwords.append(2)
        text = review["reviewText"]
        stars = review["overall"]
        categories = getUniqueCategoriesSingle(metaData[review["asin"]])
        splitted = text.split(" ")
        splitted.insert(0, "#None")
        if(splitted==None):
            continue
        for i,word in enumerate(splitted):
            index = worddic.index(word)
            lastwords.append(index)
            lastwords.pop(0)

            #generate categories vector
            catFeatures = deepcopy(listNullCats)
            for cat in categories:
                idx = uniqueCats.index(cat)
                catFeatures[idx] = 1.0

            #concat our features to one vector
            if(i==len(splitted)-1):
                #eot
                sample = catFeatures + lastwords + [1 ,stars]
                labels.append(1)
            else:
                sample = catFeatures + lastwords + [0 ,stars]
                next_word = splitted[i+1]
                index = worddic.index(next_word)
                labels.append(index)
            samples.append(np.array(sample))
    #print("Done!")
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

if __name__ == '__main__':
    print("reading meta data ...")
    metaData = dataToDict("data/music/meta_Digital_Music.json.gz")
    print("Done!")
    print("reading review Data ...")
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz", 0, v_sampleReviewSize)
    revLen = len(reviewsData)
    print("Done!")
    print("unification of categories ...")
    uniqueCats = getUniqueCategories(metaData)
    print("Done!")
    lenUniqueCats = len(uniqueCats)
    listNullCats = []
    for i in range(lenUniqueCats):
        listNullCats.append(0)
    worddic= ["#beginningOfText", "#endOfText", "#None"]

    print("creating worddic...")
    for epoch, review in enumerate(reviewsData):
        if (epoch % v_feedbackFrequency == 0):
            print(str(epoch)+"/"+str(revLen))
        text = review["reviewText"]
        splitted = text.split(v_splitString)
        for word in splitted:
            if word not in worddic:
                worddic.append(word)
    print("Done! Created worddic with size " + str(len(worddic)))

    #######

    clf = generateClassifier()

    for i in range (int(v_sampleReviewSize / v_samplePackageSize)+1):
        start = i * v_samplePackageSize
        end = min((i+1)*v_samplePackageSize, v_sampleReviewSize)
        if (start == end):
            break
        samples, labels = generateSamples(start, end, reviewsData, metaData, worddic)
        trainClassifier(clf, np.array(samples), np.array(labels), len(worddic))


    saveClassifier(clf, v_version)
    #clf2 = loadClassifier(v_version)
    print("test prediction...")
    testsamples, testlabel = generateSamples(1, 2, reviewsData, metaData, worddic)
    labelidx = clf.predict(np.array([testsamples[0]]))
    print("Done!")
    print(labelidx, worddic[labelidx[0]])

    #uniqueCategories = getUniqueCategories(metaData)
    #uniqueBrandNames = getUniqueBrandNames(metaData)

    #print(metaData)
    #imageFeatureData = open("data/baby/image_features_Baby.b", "rb")
