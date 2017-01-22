# a complete code for iris data from selection,cluster,clustering,to ensemble
from sklearn import cluster

import numpy as np
import pandas as pd
import dataSetPreprocessing as dataPre


def KMeans_c(dataSet):
    """
    KMeans directly
    :param dataSet:
    :return:
    """
    print ('KMeans_c:')
    n_c = dataPre.rand.randint(1, 10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(dataSet)
    return feature_pred


def FS_c(dataSet, r_clusters):
    """
    random feature selection before clustering by KMeans
    :param dataSet:
    :param r_clusters:
    :return:
    """
    print ('FS_c:')
    sample = dataPre.RepetitionRandomSampling(dataSet, dataSet.shape[1])
    clf = cluster.KMeans(n_clusters=r_clusters)
    feature_pred = clf.fit_predict(sample)
    return feature_pred


def RSNC_c(dataSet, target, r_clusters=3, r_state=50, r_SSR=0.7):
    """
    random selection before clustering by KMeans and then put those unselected data to the nearest centroids
    :param dataSet:
    :param target:
    :param r_clusters:
    :param r_state:
    :return:
    """
    dataSet = pd.DataFrame(dataSet)               #add index to dataSet
    target = pd.DataFrame(target)                 #add index to target
    data_selected, data_unselected, target_selected, target_unselected = dataPre.train_test_split(dataSet, target, train_size=r_SSR, random_state=r_state)
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected.values)
    result_selected = target_selected.copy()      #to clear up the copy warning
    result_selected['1'] = clf.labels_

    target_unselected_pred = []
    # convert dataframe to ndarray
    duv = np.array(data_unselected)
    length = clf.cluster_centers_.shape[0]
    dimension_count = clf.cluster_centers_.shape[1]

    # matrix operations instead of for-loop and apply_along_axis
    for urow in duv:
        tiled = np.tile(urow, (length, 1))
        diff = clf.cluster_centers_ - tiled
        diff = diff ** 2
        # column-vector of all ones
        ones = np.ones((dimension_count, 1))
        matrixrowsum = np.dot(diff, ones)
        # get the index of the smallest distance (i.e, nearest sample)
        minTag = np.argmin(matrixrowsum)
        target_unselected_pred.append(minTag)


    # bad practice for efficiency
    # target_unselected_pred = []
    # for i in range(len(data_unselected.values)):
    #     minDist = np.inf
    #     minIndex = -1
    #     for j in range(len(clf.cluster_centers_)):
    #         if distEclud(data_unselected.values[i],clf.cluster_centers_[j]) < minDist:
    #             minDist = distEclud(data_unselected.values[i],clf.cluster_centers_[j])
    #             minIndex = j
    #     target_unselected_pred.append(minIndex)

    result_unselected =  target_unselected.copy()
    result_unselected['1'] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))           #reindex

    print ('The output labels after RSNC_c:\n')
    return result['1'].values


def RSNN_c(dataSet, target, r_clusters=3, r_state=50, r_SSR=0.7):
    """
    random selection before clustering by KMeans
    then put those unselected data to the centroids which its nearest sample belongs to
    :param dataSet:
    :param target:
    :param r_clusters:
    :param r_state:
    :param r_SSR:
    :return:
    """
    dataSet = pd.DataFrame(dataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, target_selected, target_unselected = dataPre.train_test_split(dataSet, target, train_size=r_SSR, random_state=r_state)
    clf = cluster.KMeans(n_clusters=r_clusters)
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
    result = result.reindex(range(0, len(target)))            # reindex
    print ('The output labels after RSNN_c:\n')
    return result['1'].values


def FSRSNN_c(dataSet, target, r_clusters=3, r_state=50, r_FSR=0.7, r_SSR=0.7):
    """
    first random feature selection
    then random selection before clustering by KMeans
    then put those unselected data to the centroids which its nearest sample belongs to
    :param dataSet:
    :param target:
    :param r_clusters:
    :param r_state:
    :param r_FSR
    :param r_SSR
    :return:
    """
    n_FSR = int(r_FSR * dataSet.shape[1])
    featureSampledDataSet = dataPre.RepetitionRandomSampling(dataSet, n_FSR)
    print ('After feature Sampling,there remains %d features.' %featureSampledDataSet.shape[1])

    featureSampledDataSet = pd.DataFrame(featureSampledDataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, target_selected, target_unselected = dataPre.train_test_split(featureSampledDataSet, target,
                                                                                          train_size=r_SSR,
                                                                                          random_state=r_state)
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected[1] = clf.labels_

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
    result_unselected[1] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))  # reindex
    print ('The output labels after FSRSNN_c:\n')
    # print result
    return result[1].values




def Test():
    """
        try push
    """
    # test for RSNN_c
    dataSet, target = dataPre.loadIris()
    r_clusters = dataPre.rand.randint(2, 10)
    r_state = dataPre.rand.randint(1, 100)
    result = RSNN_c(dataSet, target, r_clusters, r_state)
    print (result)

    #test for FSRSNN_c
    #dataSet, target = loadIris()
    #result = FSRSNN_c(dataSet, target)
    #print result

# Test()




