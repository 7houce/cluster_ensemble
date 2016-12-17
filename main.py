# a complete code for iris data from selection,cluster,clustering,to ensemble
from sklearn import datasets
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

import random as rand
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# simply load the iris data
def loadIris():
    print 'load Iris:'
    iris = datasets.load_iris()
    return iris.data,iris.target

# load the common dataSet (labels in the last column)
def loadDataSet():
    dataSet = []
    target = []
    fr = open('F:\\UCI Data\\wine\\wine.data.txt')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(int(curLine[0]))
        curLine.remove(curLine[0])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)
    print dataSet
    return np.array(dataSet),np.array(target)

# repetition random sampling by features
def RepetitionRandomSampling(dataSet, n):
    print 'RepetitionRandomSampling:'
    sample = []
    dataSet = dataSet.T
    for i in range(n):
        sample.append(dataSet[rand.randint(0,len(dataSet)-1)])
    #sample = np.array(sample)  # change list to np.array  with it or not is ok,for list can be directly converted into dataFrame type
    dataFrame = pd.DataFrame(sample)
    sample = dataFrame.drop_duplicates()
    sample = np.array(sample)
    return sample.T

# calculate the distance of the two samples
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA-vecB,2)))

# directly clustering by KMeans
def KMeans_c(dataSet):
    print 'KMeans_c:'
    n_c = rand.randint(1,10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(dataSet)
    return feature_pred

# random feature selection before clustering by KMeans
def FS_c(dataSet):
    print 'FS_c:'
    sample = RepetitionRandomSampling(dataSet, dataSet.shape[1])
    n_c = rand.randint(1, 10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(sample)
    return feature_pred

# random selection before clustering by KMeans and then put those unselected data to the nearest centroids
def RSNC_c(dataSet, target):
    print 'RSNC_c:'
    n_c = rand.randint(1,10)
    r_s = rand.randint(1,100)
    dataSet = pd.DataFrame(dataSet)       #add index to dataSet
    target = pd.DataFrame(target)       #add index to target
    data_selected, data_unselected, target_selected, target_unselected = train_test_split(dataSet,target,train_size=0.7,random_state=r_s)
    clf = cluster.KMeans(n_clusters=n_c)
    clf.fit(data_selected.values)
    result_selected = target_selected.copy()      #to clear up the copy warning
    result_selected['1'] = clf.labels_

    target_unselected_pred = []
    for i in range(len(data_unselected.values)):
        minDist = np.inf
        minIndex = -1
        for j in range(len(clf.cluster_centers_)):
            if distEclud(data_unselected.values[i],clf.cluster_centers_[j]) < minDist:
                minDist = distEclud(data_unselected.values[i],clf.cluster_centers_[j])
                minIndex = j
        target_unselected_pred.append(minIndex)

    result_unselected =  target_unselected.copy()
    result_unselected['1'] = np.array(target_unselected_pred)
    result = pd.concat([result_selected,result_unselected])
    result = result.reindex(range(0,len(target)))         #reindex
    return result['1'].values

# random selection before clustering by KMeans and then put those unselected data to the centroids which its nearest sample belongs to
def RSNN_c(dataSet, target):
    print 'RSNN_c:'
    n_c = rand.randint(1,10)
    r_s = rand.randint(1,100)
    dataSet = pd.DataFrame(dataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, target_selected, target_unselected = train_test_split(dataSet,target,train_size=0.7,random_state=r_s)
    clf = cluster.KMeans(n_clusters=n_c)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected['1'] = clf.labels_

    target_unselected_pred = []
    for i in range(len(data_unselected.values)):
        minDist = np.inf
        minIndex = -1
        for j in range(len(data_selected.values)):
            if distEclud(data_unselected.values[i],data_selected.values[j]) < minDist:
                minDist = distEclud(data_unselected.values[i],data_selected.values[j])
                minIndex = j
        target_unselected_pred.append(clf.labels_[minIndex])

    result_unselected = target_unselected.copy()
    result_unselected['1'] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))  # reindex
    return result['1'].values

# test
def Test():
    dataSet, target = loadIris()
    result = RSNN_c(dataSet, target)
    print result

Test()




