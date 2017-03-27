"""
Single Ensemble Member Generation.
Author : Zhijie Lin, Ce Zhou.
"""
from __future__ import print_function
from sklearn import cluster
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cross_validation import train_test_split


def _euclidean_distance_decider(duv, centers, labels, dsv, type='center'):
    """
    Decide the labels of unselected samples by pure euclidean distances calculation
    (internal use only)
    """
    target_unselected_pred = []
    if type == 'center':
        all_distances = euclidean_distances(centers, duv, squared=True)
    else:
        all_distances = euclidean_distances(dsv, duv, squared=True)
    for i in range(0, len(duv)):
        min_index = np.argmin(all_distances[:, i])
        target_unselected_pred.append(min_index if type == 'center' else labels[min_index])
    return target_unselected_pred


def _nearest_neighbor_decider(duv, centers, labels, dsv, type='center'):
    """
    Decide the labels of unselected samples by k-nearest neighbor
    (internal use only)
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(centers if type == 'center' else dsv)
    target_unselected_pred = []
    for urow in duv:
        _, index = neigh.kneighbors(urow)
        target_unselected_pred.append(index if type == 'center' else labels[int(index)])
    return target_unselected_pred


_decider = {'euclidean': _euclidean_distance_decider,
            'knn': _nearest_neighbor_decider}


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


def FS(dataset, n_fsr, n_clusters):
    """
    perform K-Means on the feature-selected dataset (FS)
    (Used for generating base-solutions)

    Parameters
    ----------
    :param dataset: dataset to be clustered
    :param n_fsr:
    :param n_clusters: num of clusters

    Returns
    -------
    :return: predicted labels
    """
    print ('===[Feature Sampling Member Generation]===')
    sample = feature_sampling(dataset, n_fsr)
    clf = cluster.KMeans(n_clusters=n_clusters)
    feature_pred = clf.fit_predict(sample)
    return feature_pred


def RSNC(dataset, target, r_clusters=3, r_state=50, r_SSR=0.7):
    """
    random instances sampling before clustering by KMeans
    unselected instances determination : nearest centroids

    :param dataset:
    :param target:
    :param r_clusters:
    :param r_state:
    :return:
    """
    # use pandas to reindex the sampled-out instances
    dataset = pd.DataFrame(dataset)
    target = pd.DataFrame(target)
    data_selected, data_unselected, target_selected, target_unselected = \
        train_test_split(dataset, target, train_size=r_SSR, random_state=r_state)
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected.values)
    result_selected = target_selected.copy()
    result_selected[1] = clf.labels_

    target_unselected_pred = []
    duv = np.array(data_unselected)

    all_distances = euclidean_distances(clf.cluster_centers_, duv, squared=True)
    for i in range(0, len(duv)):
        minIndex = np.argmin(all_distances[:, i])
        target_unselected_pred.append(minIndex)

    result_unselected =  target_unselected.copy()
    result_unselected['1'] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))           #reindex

    return result[1].values


def RSNN(dataSet, target, r_clusters=3, r_state=50, r_SSR=0.7):
    """
    random instances sampling before clustering by KMeans
    unselected instances determination : nearest neighbors

    :param dataSet:
    :param target:
    :param r_clusters:
    :param r_state:
    :param r_SSR:
    :return:
    """
    dataSet = pd.DataFrame(dataSet)
    target = pd.DataFrame(target)
    data_selected, data_unselected, target_selected, target_unselected = \
        train_test_split(dataSet, target, train_size=r_SSR, random_state=r_state)
    clf = cluster.KMeans(n_clusters=r_clusters)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected['1'] = clf.labels_

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

    result_unselected = target_unselected.copy()
    result_unselected['1'] = np.array(target_unselected_pred)
    result = pd.concat([result_selected, result_unselected])
    result = result.reindex(range(0, len(target)))            # reindex
    return result['1'].values


def FSRSNN(dataSet, target, r_clusters=3, r_state=50, fsr=0.7, ssr=0.7):
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
    target_selected, target_unselected = train_test_split(featureSampledDataSet, target,
                                                                  train_size=n_SSR,
                                                                  random_state=r_state)
    t2 = time.clock()
    sampling_time = t2 - t1

    t1 = time.clock()
    # perform K-Means on double-sampled dataset with given k
    if r_clusters >= len(data_selected):
        r_clusters = len(data_selected) / 2
    clf = cluster.KMeans(n_clusters=r_clusters, n_init=1)
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


def FSRSNC(dataSet, target, r_clusters=3, r_state=50, fsr=0.7, ssr=0.7, decider='euclidean'):
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
        feature_left_amount = int(fsr)
    elif 0 < fsr <= 1.0:
        feature_left_amount = int(fsr * dataSet.shape[1])
    else:
        raise ValueError('FSRSNN : fsr should be a positive value!')

    # convert ssr to absolute number
    if ssr > 1.0:
        sample_left_amount = int(ssr)
    elif 0 < ssr <= 1.0:
        sample_left_amount = int(ssr * dataSet.shape[0])
    else:
        raise ValueError('FSRSNN : ssr should be a positive value!')

    # perform feature sampling
    feature_sampled_data = feature_sampling(dataSet, feature_left_amount)
    print ('After feature Sampling,there remains %d features.' % feature_sampled_data.shape[1])

    # perform instance sampling
    # we use pandas's index to merge the unselected instances after
    feature_sampled_data = pd.DataFrame(feature_sampled_data)
    target = pd.DataFrame(target)
    data_selected, data_unselected, \
    target_selected, target_unselected = train_test_split(feature_sampled_data, target,
                                                          train_size=sample_left_amount,
                                                          random_state=r_state)
    t2 = time.clock()
    sampling_time = t2 - t1

    t1 = time.clock()
    # perform K-Means on double-sampled dataset with given k
    if r_clusters >= len(data_selected):
        r_clusters = len(data_selected) / 2
    clf = cluster.KMeans(n_clusters=r_clusters, n_init=1)
    clf.fit(data_selected)
    result_selected = target_selected.copy()
    result_selected[1] = clf.labels_
    t2 = time.clock()
    clustering_time = t2 - t1

    t1 = time.clock()
    # determine the labels of those unselected by NN strategy
    duv = np.array(data_unselected)
    dsv = np.array(data_selected)
    target_unselected_pred = _decider[decider](duv, clf.cluster_centers_, clf.labels_, dsv, type='center')

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
