import json
import gzip

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

#######################
# EXTRACT FEATURES
#######################

def getUniqueCategories(metaData):
    output = set([])
    for item in metaData:
        for it1 in item["categories"]:
            for it2 in it1:
                output.add(it2)
    return list(output)

def getUniqueBrandNames(metaData):
    output = set([])
    i=0
    for item in metaData:
        try:
            output.add(item["brand"])
        except:
            i+=1
    print(i)
    return list(output)

if __name__ == '__main__':
    metaData = dataToList("data/music/meta_Digital_Music.json.gz")
    #reviewsData = dataToList("data/baby/reviews_Baby.json.gz")
    #uniqueCategories = getUniqueCategories(metaData)
    #uniqueBrandNames = getUniqueBrandNames(metaData)

    print(metaData[0])
    #imageFeatureData = open("data/baby/image_features_Baby.b", "rb")
