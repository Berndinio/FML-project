import json
import gzip
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB
import numpy as np
#######################
# DATA LOADING
#######################
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def dataToList(path):
    metaData = []
    for item in parse(path):
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

if __name__ == '__main__':
    print("reading meta data ...")
    metaData = dataToDict("data/music/meta_Digital_Music.json.gz")
    print("Done!")
    print("reading review Data ...")
    reviewsData = dataToList("data/music/reviews_Digital_Music.json.gz")
    reviewsData = reviewsData[0:1000]
    revLen = len(reviewsData)
    print("Done!")
    print("unification of categories ...")
    uniqueCats = getUniqueCategories(metaData)
    print("Done!")
    lenUniqueCats = len(uniqueCats)
    listNullCats = []
    for i in range(lenUniqueCats):
        listNullCats.append(0)

    samples = []
    labels = []
    worddic= ["#beginningOfText", "#endOfText", "#None"]
    print("starting samples generating...")
    for epoch, review in enumerate(reviewsData):
        if (epoch % 100 == 0):
            print(str(epoch)+"/"+str(revLen))
        lastwords = [2, 2, 2, 2, 2]
        text = review["reviewText"]
        stars = review["overall"]
        categories = getUniqueCategoriesSingle(metaData[review["asin"]])
        splitted = text.split(" ")
        splitted.insert(0, "#None")
        if(splitted==None):
            continue
        for i,word in enumerate(splitted):
            if word not in worddic:
                worddic.append(word)
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
                if next_word not in worddic:
                    worddic.append(next_word)
                index = worddic.index(next_word)
                labels.append(index)
            samples.append(np.array(sample))
    print("Done!")
    print("starting training...")
    samples = np.array(samples)

    clf = MultinomialNB()
    clf.fit(np.array(samples), np.array(labels))
    print("Done!")
    print("test prediction...")
    labelidx = clf.predict(np.array([samples[0]]))
    print("Done!")
    print(labelidx, worddic[labelidx[0]])
    print(samples[0], labels[0])
    print(samples[1], labels[1])
    print(samples[2], labels[2])

    #uniqueCategories = getUniqueCategories(metaData)
    #uniqueBrandNames = getUniqueBrandNames(metaData)

    #print(metaData)
    #imageFeatureData = open("data/baby/image_features_Baby.b", "rb")
