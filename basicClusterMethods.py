# a complete code for iris data from selection,cluster,clustering,to ensemble
from __future__ import print_function
from sklearn import cluster
import numpy as np
import pandas as pd
import dataSetPreprocessing as dataPre
import time
from sklearn.neighbors import NearestNeighbors


def feature_sampling(dataset, n, replacement=True):
    """
    perform feature sampling on given dataset

    Parameters
    ----------
    :param dataset: dataset to perform feature sampling
    :param n: number of features left
    :param replacement: sampling with replacement or not

    Returns
    -------
    :return: sampled dataset
    """
    n_feature = dataset.shape[1]
    print ('feature sampling....')
    if n <= 0:
        raise ValueError('feature sampling : fsr should be a positive value!')
    elif n > n_feature:
        raise ValueError('feature sampling : fsr should be lesser than n_feature!')
    indices = np.random.choice(n_feature, n, replace=replacement)
    indices = np.unique(indices)
    return dataset[:, indices]


def KMeans_c(dataSet):
    """
    perform K-Means to the given dataset directly
    (Used for generating base-solutions)

    Parameters
    ----------
    :param dataSet: dataset to be clustered

    Returns
    -------
    :return: predicted labels
    """
    print ('KMeans_c:')
    n_c = dataPre.rand.randint(1, 10)
    clf = cluster.KMeans(n_clusters=n_c)
    feature_pred = clf.fit_predict(dataSet)
    return feature_pred


def FS_c(dataSet, r_clusters):
    """
    perform K-Means on the feature-selected dataset (FS)
    (Used for generating base-solutions)

    Parameters
    ----------
    :param dataSet: dataset to be clustered
    :param r_clusters: num of clusters

    Returns
    -------
    :return: predicted labels
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


def FSRSNN_c(dataSet, target, r_clusters=3, r_state=50, fsr=0.7, ssr=0.7):
    """
    perform double-sampling (nearest neighbor version) on the dataset,
    and then, perform K-Means on the sampled dataset
    (Used for generating base-solutions)

    Parameters
    ----------
    :param dataSet: dataset to be clustered
    :param target:  real label of given dataset
    :param r_clusters: number of clusters
    :param r_state: random state
    :param fsr: feature sampling rate (if <= 1.0) or absolute count (if > 1.0)
    :param ssr: instances sampling rate (if <= 1.0) or absolute count (if > 1.0)

    Returns
    -------
    :return: predicted labels
    """
    t1 = time.clock()
    # convert fsr to absolute number
    if fsr > 1.0:
        n_FSR = int(fsr)
    elif 0 < fsr <= 1.0:
        n_FSR = int(fsr * dataSet.shape[1])
    else:
        raise ValueError('FSRSNN : fsr should be a positive value!')

    # convert ssr to absolute number
    if ssr > 1.0:
        n_SSR = int(ssr)
    elif 0 < ssr <= 1.0:
        n_SSR = int(ssr * dataSet.shape[0])
    else:
        raise ValueError('FSRSNN : ssr should be a positive value!')

    # perform feature sampling
    featureSampledDataSet = feature_sampling(dataSet, n_FSR)
    print ('After feature Sampling,there remains %d features.' %featureSampledDataSet.shape[1])

    # perform instance sampling
    # we use pandas's index to merge the unselected instances after
    featureSampledDataSet = pd.DataFrame(featureSampledDataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, \
    target_selected, target_unselected = dataPre.train_test_split(featureSampledDataSet, target,
                                                                  train_size=n_SSR,
                                                                  random_state=r_state)
    t2 = time.clock()
    sampling_time = t2 - t1

    t1 = time.clock()
    # perform K-Means on double-sampled dataset with given k
    if r_clusters >= len(data_selected):
        r_clusters = len(data_selected) / 2
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected[1] = clf.labels_
    t2 = time.clock()
    clustering_time = t2 - t1

    t1 = time.clock()
    # determine the labels of those unselected by NN strategy
    target_unselected_pred = []
    duv = np.array(data_unselected)
    dsv = np.array(data_selected)

    # use sklearn's NearestNeighbors to find nearest instance
    # kd-tree / balltree will obtain a good performance far from pure matrix operations
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dsv)
    for urow in duv:
        _, index = neigh.kneighbors(urow)
        target_unselected_pred.append(clf.labels_[int(index)])

    # merge the unselected instances with selected instances in true order
    result_unselected = target_unselected.copy()
    result_unselected[1] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))
    t2 = time.clock()
    nn_time = t2 - t1

    print ('FSRSNN completed.')
    print ('Sampling time = ' + str(sampling_time))
    print ('Clustering time = ' + str(clustering_time))
    print ('Nearest Neighbor Determination Time = ' + str(nn_time))
    print ('Data Selected Matrix Size = ' + str(dsv.shape))
    print ('Data Unselected Matrix Size = ' + str(duv.shape))
    print ('=================================================================')

    return result[1].values


def FSRSNC_c(dataSet, target, r_clusters=3, r_state=50, fsr=0.7, ssr=0.7):
    """
    perform double-sampling (nearest centroid version) on the dataset,
    and then, perform K-Means on the sampled dataset
    (Used for generating base-solutions)

    Parameters
    ----------
    :param dataSet: dataset to be clustered
    :param target:  real label of given dataset
    :param r_clusters: number of clusters
    :param r_state: random state
    :param fsr: feature sampling rate (if <= 1.0) or absolute count (if > 1.0)
    :param ssr: instances sampling rate (if <= 1.0) or absolute count (if > 1.0)

    Returns
    -------
    :return: predicted labels
    """
    t1 = time.clock()
    # convert fsr to absolute number
    if fsr > 1.0:
        n_FSR = int(fsr)
    elif 0 < fsr <= 1.0:
        n_FSR = int(fsr * dataSet.shape[1])
    else:
        raise ValueError('FSRSNN : fsr should be a positive value!')

    # convert ssr to absolute number
    if ssr > 1.0:
        n_SSR = int(ssr)
    elif 0 < ssr <= 1.0:
        n_SSR = int(ssr * dataSet.shape[0])
    else:
        raise ValueError('FSRSNN : ssr should be a positive value!')

    # perform feature sampling
    featureSampledDataSet = feature_sampling(dataSet, n_FSR)
    print ('After feature Sampling,there remains %d features.' %featureSampledDataSet.shape[1])

    # perform instance sampling
    # we use pandas's index to merge the unselected instances after
    featureSampledDataSet = pd.DataFrame(featureSampledDataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, \
    target_selected, target_unselected = dataPre.train_test_split(featureSampledDataSet, target,
                                                                  train_size=n_SSR,
                                                                  random_state=r_state)
    t2 = time.clock()
    sampling_time = t2 - t1

    t1 = time.clock()
    # perform K-Means on double-sampled dataset with given k
    if r_clusters >= len(data_selected):
        r_clusters = len(data_selected) / 2
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected[1] = clf.labels_
    t2 = time.clock()
    clustering_time = t2 - t1

    t1 = time.clock()
    # determine the labels of those unselected by NN strategy
    target_unselected_pred = []
    duv = np.array(data_unselected)
    dsv = np.array(data_selected)

    # use sklearn's NearestNeighbors to find nearest instance
    # kd-tree / balltree will obtain a good performance far from pure matrix operations
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(clf.cluster_centers_)
    for urow in duv:
        _, index = neigh.kneighbors(urow)
        target_unselected_pred.append(int(index))

    # merge the unselected instances with selected instances in true order
    result_unselected = target_unselected.copy()
    result_unselected[1] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))
    t2 = time.clock()
    nn_time = t2 - t1

    print ('FSRSNC completed.')
    print ('Sampling time = ' + str(sampling_time))
    print ('Clustering time = ' + str(clustering_time))
    print ('Nearest Centroid Determination Time = ' + str(nn_time))
    print ('Data Selected Matrix Size = ' + str(dsv.shape))
    print ('Data Unselected Matrix Size = ' + str(duv.shape))
    print ('=================================================================')

    return result[1].values
