# a complete code for iris data from selection,cluster,clustering,to ensemble
from sklearn import datasets
from sklearn import cluster
from sklearn.cross_validation import train_test_split

import random as rand
import numpy as np
import pandas as pd


def loadIris():
    """
    simply load the iris data
    :return: tuple(data, target)
    """
    print 'load Iris:'
    iris = datasets.load_iris()
    return iris.data, iris.target


def loadDataSet():
    """
    load the common dataSet (labels in the last column)
    :return:
    """
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
    return np.array(dataSet), np.array(target)


def RepetitionRandomSampling(dataSet, n):
    """
    random sampling by features (without replacement)
    :param dataSet: dataset to be sampled
    :param n: number of features
    :return: sampled dataset (in a sub-space)
    """
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


def distEclud(vecA, vecB):
    """
    calculate the euclidean distance of 2 given vectors
    :param vecA:
    :param vecB:
    :return: euclidean distance
    """
    return np.sqrt(sum(np.power(vecA-vecB,2)))


def KMeans_c(dataSet):
    """
    KMeans directly
    :param dataSet:
    :return:
    """
    print 'KMeans_c:'
    n_c = rand.randint(1,10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(dataSet)
    return feature_pred


def FS_c(dataSet):
    """
    random feature selection before clustering by KMeans
    :param dataSet:
    :return:
    """
    print 'FS_c:'
    sample = RepetitionRandomSampling(dataSet, dataSet.shape[1])
    n_c = rand.randint(1, 10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(sample)
    return feature_pred


def RSNC_c(dataSet, target):
    """
    random selection before clustering by KMeans and then put those unselected data to the nearest centroids
    :param dataSet:
    :param target:
    :return:
    """
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

def RSNN_c(dataSet, target):
    """
    random selection before clustering by KMeans
    then put those unselected data to the centroids which its nearest sample belongs to
    :param dataSet:
    :param target:
    :return:
    """
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
    # convert dataframe to ndarray
    duv = np.array(data_unselected)
    dsv = np.array(data_selected)
    length = dsv.shape[0]
    dimension_count = dsv.shape[1]
    # matrix operations instead of for-loop and apply_along_axis
    for urow in duv:
        tiled = np.tile(urow, (length, 1))
        diff = dsv - tiled
        diff = diff ** 2
        # column-vector of all ones
        ones = np.ones((dimension_count, 1))
        matrixrowsum = np.dot(diff, ones)
        # get the index of the smallest distance (i.e, nearest sample)
        minTag = np.argmin(matrixrowsum)
        target_unselected_pred.append(clf.labels_[minTag])

    # bad practice for efficiency
    # target_unselected_pred = []
    # for i in range(len(data_unselected.values)):
    #     minDist = np.inf
    #     minIndex = -1
    #     for j in range(len(data_selected.values)):
    #         if distEclud(data_unselected.values[i],data_selected.values[j]) < minDist:
    #             minDist = distEclud(data_unselected.values[i],data_selected.values[j])
    #             minIndex = j
    #     target_unselected_pred.append(clf.labels_[minIndex])

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




