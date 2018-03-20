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
### MAIN VARIABLES

# how much words are viewed at during classification
v_sequenceLength = 3
# how many reviews should be used for training?
v_sampleReviewSize = 257527
# how many reviews should be trained at once? (lower if RAM is overloaded)
v_samplePackageSize = 10
# which timestamps should be saved and feedbacked?
v_feedbackFrequency = 5
# savings folder in use
v_version = 1

### TRAINING TEXT MANIPULATION
v_splitString = " "
v_replaceInString = ["&quot;", "\"", "(", ")", ";"]
v_learningAlgorithm = "" #not implemented yet
v_messageStart = "#beginningOfText"
v_messageEnd = "#endOfText"
v_chooseTrainingRatingRangeStart = 5.0
v_chooseTrainingRatingRangeEnd = 5.0
v_chooseTrainingWordsRangeStart = 1
v_chooseTrainingWordsRangeEnd = 150
v_chooseSimpleWords = False
v_manipulateTrainingReplace = True
v_manipulateTrainingLower = False
v_manipulateTrainingRemoveNonASCII = True
v_manipulateTrainingRemoveAdditionalWhitespaces = True

v_loadWordDic = True

### GENERATED REV VARIABLES
v_maxReviewOutputLength = 330


# FEATURES IN USE
v_activateCategories = False
v_activateRating = True
v_activateName = False # not implemented yet
v_activatePrice = True
v_activateWordcount = False

#######################
# DATA LOADING
#######################
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_basic_wordlist():
    filepath = "data/basic-word-list.txt"
    f = open(filepath, 'r')
    raw_wordlist = str(f.read())
    for replacement in [".", "\n", ",", ":", "(", ")", "Basic", "Intl", "Next", "Compound", "Endings"]:
        raw_wordlist = raw_wordlist.replace(replacement, "")
    raw_wordlist = '\n'.join(' '.join(line.split()) for line in raw_wordlist.splitlines())
    wordlist = raw_wordlist.split(" ")
    for i in range(len(wordlist)):
        wordlist[i] = wordlist[i].lower()
    wordlist = set(wordlist)
    wordlist = list(wordlist)
    wordlist.sort()
    print("Basic wordlist loaded!")
    return wordlist

def dataToListOld(path, start, end):
    metaData = []
    for counter, item in enumerate(parse(path)):
        if (counter >= end):
            return metaData
        if (counter >= start):
            metaData.append(item)
    return metaData

def dataToList(path, start, end, worddic):
    data = []
    counter = 0
    final_counter = 0
    jumped = 0
    for item in parse(path):
        if (final_counter >= end):
            return data, jumped
        if (counter <= start):
            counter += 1
            continue
        jumped +=1
        if (item["overall"] >= v_chooseTrainingRatingRangeStart) and (item["overall"] <= v_chooseTrainingRatingRangeEnd):
            if (len(item["reviewText"]) >= v_chooseTrainingWordsRangeStart) and (len(item["reviewText"]) <= v_chooseTrainingWordsRangeEnd):
                item["reviewText"] = manipulateText(item["reviewText"])
                if (v_chooseSimpleWords):
                    simple = True
                    text = item["reviewText"]
                    splitted = text.split(" ")
                    for word in splitted:
                        word = word.lower()
                        if (word not in worddic) and (word not in v_replaceInString):
                            simple = False
                            break
                    if simple:
                        final_counter +=1
                        data.append(item)
                else:
                    final_counter +=1
                    data.append(item)

    print("Completely iterated over dataset")
    return data, jumped

def manipulateText(text):
    if v_manipulateTrainingRemoveNonASCII:
        text = ''.join([x for x in text if ord(x) < 128])
    if v_manipulateTrainingReplace:
        for replacement in v_replaceInString:
            text = text.replace(replacement, "")
    if v_manipulateTrainingRemoveAdditionalWhitespaces:
        text = '\n'.join(' '.join(line.split()) for line in text.splitlines())
    if v_manipulateTrainingLower:
        text = text.lower()
    return text

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
        if "categories" in item.keys():
            for it1 in item["categories"]:
                for it2 in it1:
                    output.add(it2)
    return sorted(list(output))

def getUniqueCategoriesSingle(item):
    output = set([])
    if "categories" in item.keys():
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

def saveWorddic(worddic, filename ,version=0):
    print("saving worddic to file, version = " + str(version))
    saveObject(worddic, "worddic/"+filename, version)

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

def generateFeatures(featurelist, lastwords, catFeatures, stars, price, name, counter):
    categories_is_active = featurelist[0]
    rating_is_active = featurelist[1]
    title_is_active = featurelist[2]
    price_is_active = featurelist[3]
    wordcount_is_active = featurelist[4]

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
    if wordcount_is_active:
        sample.append(counter)
    sample.extend(lastwords)
    return sample

def trainClassifier(classifier, samples, labels, worddicSize):
    print("Training classifier...")
    classifier.partial_fit(samples, labels, range(worddicSize))
    print("Done partial training!")

def generateSamples(reviewsData, metaData, worddic, featurelist, iteration):
    global uniqueCats
    global listNullCats
    print("Called generate samples with " + str(len(reviewsData)) + " reviews.")

    samples = []
    labels = []
    for epoch, review in enumerate(reviewsData):
        if (((iteration*v_samplePackageSize) +epoch) % v_feedbackFrequency == 0):
            print("samples generated:" + str(((iteration*v_samplePackageSize) +epoch))+"/"+str(v_sampleReviewSize))
        lastwords = []
        for i in range(v_sequenceLength):
            lastwords.append(2)

        if not "reviewText" in review.keys():
            print("'ReviewText' missing")
            review_text = ""
        else:
            review_text = review["reviewText"]

        if not "overall" in review.keys():
            print("'Overall' missing")
            stars = 0.0
        else:
            stars = review["overall"]
        #usually text manipulated here
        product = review["asin"]

        if not "title" in metaData[product].keys():
            print("'Title' missing")
            name = ""
        else:
            name = metaData[product]["title"]

        if not "price" in metaData[product].keys():
            print("'Price' missing")
            price = 0.0
        else:
            price = metaData[product]["price"]

        categories = getUniqueCategoriesSingle(metaData[review["asin"]])
        splitted = review_text.split(" ")
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
                #label is #EndOfText
                labels.append(1)
            else:
                next_word = splitted[i+1]
                index = worddic.index(next_word)
                labels.append(index)
            sample = generateFeatures(featurelist, lastwords, catFeatures, stars, price, name, i)
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
    for i in range(v_sequenceLength):
        lastwords.append(2)

    if not all(k in product.keys() for k in ["categories", "overall", "price", "title"]):
        return "This is an error text...no really...there was a key missing."
    #extracting product features
    categories = product["categories"]
    stars = product["overall"]
    price = product["price"]
    name = product["title"]

    #generate categories vector
    catFeatures = deepcopy(listNullCats)
    for cat in categories:
        idx = uniqueCats.index(cat)
        catFeatures[idx] = 1.0

    text = [worddic[0]]
    while (text[-1]!=worddic[1] and len(text) < v_maxReviewOutputLength):
        sample = generateFeatures(featurelist, lastwords, catFeatures, stars, price, name, len(text))
        prediction = classifier.predict_proba(np.array([np.array(sample)]))
        idx = np.argmax(prediction[0])
        word = worddictionary[idx]
        text.append(word)
        lastwords.pop(0)
        lastwords.append(idx)
    return text

#######################
# MAIN LOOP
#######################

if __name__ == '__main__':


    global uniqueCats
    global listNullCats

    jumping_point = 0
    worddic= [v_messageStart, v_messageEnd, "#None"]
    if v_chooseSimpleWords:
        worddic.extend(load_basic_wordlist())

    # some loading
    print("reading meta data ...")
    metaData = dataToDict("data/music/meta_Digital_Music.json.gz")
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
    print("reading review Data ...")
    reviewsData, trash = dataToList("data/music/reviews_Digital_Music.json.gz", 0, 836006, worddic)
    print("Dataset has length "+str(len(reviewsData)))
    worddic = []
    if not v_chooseSimpleWords:
        if not v_loadWordDic:
            print("creating worddic...")
            for epoch, review in enumerate(reviewsData):
                if (epoch % v_feedbackFrequency == 0):
                    print(str(epoch)+"/"+str(836006))
                text = review["reviewText"]
                splitted = text.split(v_splitString)
                worddic = worddic + splitted
                if(epoch%100 == 0):
                    worddic = list(set(worddic))
            saveObject(worddic, "worddic/words.nb", v_version)
        else:
            worddic = loadObject("worddic/words.nb", v_version)
            print("Loaded Worddic")
        print("Done! Created worddic with size " + str(len(worddic)))

    # chosen Classifier
    clf = generateClassifier()
    sampletype = [v_activateCategories, v_activateRating, v_activateName, v_activatePrice, v_activateWordcount]

    if(True):
        # step-wise learning
        for i in range (int(v_sampleReviewSize / v_samplePackageSize)+1):
            start = i*v_samplePackageSize
            end = min((i+1)*v_samplePackageSize, v_sampleReviewSize)
            if (i*v_samplePackageSize >= v_sampleReviewSize):
                break
            samples, labels = generateSamples(reviewsData[start:end], metaData, worddic, sampletype, i)
            try:
                trainClassifier(clf, np.array(samples), np.array(labels), len(worddic))
            except:
                print("We got an error while partial fit and ignored it.")
            #saveClassifier(clf, str(i*v_samplePackageSize)+"save.nb" , v_version)
            saveClassifier(clf, "save.nb" , v_version)



    #clf = loadClassifier("save.nb", v_version)
    print("generating test review...")
    review_product = {"categories":[uniqueCats[0]], "price":9.99, "stars":4.0, "name":"Ulala"}
    review_product = metaData[reviewsData[0]["asin"]]
    review_product["overall"] = reviewsData[0]["overall"]
    review_product["categories"] = getUniqueCategoriesSingle(review_product)
    #classifier, product, featurelist, worddictionary, randomness=False
    review_text = generateReview(clf, review_product, sampletype, worddic, False)
    text = " ".join(review_text)
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
