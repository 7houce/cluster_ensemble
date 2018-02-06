"""
Internal clustering evaluation metric
Author: Zhijie Lin
"""
from sklearn import metrics
import numpy as np
import time
import os


def _silhouette(data, label):
    return metrics.silhouette_score(data, label, metric='euclidean')


def _silhouette_instance(data, label):
    return metrics.silhouette_samples(data, label, metric='euclidean')


def _zero_one_normalizer(weights):
    max_value = np.max(weights)
    min_value = np.min(weights)
    weights -= min_value
    weights /= (max_value - min_value)
    return weights


def _zero_one_normalizer_for_dict(weights_dict):
    min_val = min(weights_dict.items(), key=lambda x: x[1])[1]
    max_val = max(weights_dict.items(), key=lambda x: x[1])[1]
    new_dict = {}
    for k, v in weights_dict.items():
        new_dict[k] = (v - min_val)/(max_val - min_val)
    return new_dict

_avail_weight_types = {'silhouette': _silhouette}
_avail_ins_weight_types = {'silhouette': _silhouette_instance}


def cal_internal_weights_for_library_as_array(data, library_name, normalize=True,
                                              library_path='Results/', weight_type='silhouette'):
    """
    calculate internal evaluation metrics as weights for cluster ensemble
    return as a ndarray

    Parameters
    ----------
    :param data: dataset
    :param library_name: name of the library (used for storing the internal metrics)
    :param normalize: normalize result to [0, 1] or not
    :param library_path: path to store internal evaluation metrics
    :param weight_type: type of internal evaluation metric, only 'silhouette' supported.

    Return
    ------
    :return: internal metrics as a 1d-ndarray
    """
    t1 = time.clock()
    dataset_name = library_name.split('_')[0]
    if os.path.isfile(library_path + dataset_name + '/' + library_name + '_internals.txt'):
        weights = np.loadtxt(library_path + dataset_name + '/' + library_name + '_internals.txt', delimiter=',')
        if normalize:
            return _zero_one_normalizer(weights)
        else:
            return weights
    labels = np.loadtxt(library_path + dataset_name + '/' + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    print '[Internal Weights] start calculation..'
    weights = []
    for label in labels:
        weights.append(_avail_weight_types[weight_type](data, label))
        print '[Internal Weights] one clustering done.'
    weights = np.array(weights)
    np.savetxt(library_path + dataset_name + '/' + library_name + '_internals.txt', weights, fmt='%.8f', delimiter=',')
    t2 = time.clock()
    print '[Internal Weights] finish calculation, time consumption:' + str(t2 - t1) + 's'
    if normalize:
        return _zero_one_normalizer(weights)
    else:
        return weights


def cal_internal_weights_for_library_cluster_as_array(data, library_name, normalize=True,
                                                      library_path='Results/', weight_type='silhouette'):
    """
    calculate internal evaluation metrics as weights for cluster ensemble
    return as a ndarray

    Parameters
    ----------
    :param data: dataset
    :param library_name: name of the library (used for storing the internal metrics)
    :param normalize: normalize result to [0, 1] or not
    :param library_path: path to store internal evaluation metrics
    :param weight_type: type of internal evaluation metric, only 'silhouette' supported.

    Return
    ------
    :return: internal metrics as a 1d-ndarray
    """
    dataset_name = library_name.split('_')[0]

    # check if the internals are already calculated, if exist, we just read them in instead of re-calculating it.
    if os.path.isfile(library_path + dataset_name + '/' + library_name + '_instance_internals.txt'):
        print '[Internal Weights] Internals are already calculated for this library.'
        weights = np.loadtxt(library_path + dataset_name + '/' + library_name + '_instance_internals.txt',
                             delimiter=',')
        labels = np.loadtxt(library_path + dataset_name + '/' + library_name + '.res', delimiter=',')
        labels = labels.astype(int)
    else:
        # calculate the internals for every base clusterings of given library in instance-level
        labels = np.loadtxt(library_path + dataset_name + '/' + library_name + '.res', delimiter=',')
        labels = labels.astype(int)
        weights = []
        print '[Internal Weights] start calculation..'
        t1 = time.clock()
        counter = 0
        for label in labels:
            counter += 1
            weights.append(_avail_ins_weight_types[weight_type](data, label))
            print '[Internal Weights] No.' + str(counter) + ' base clustering done.'
        weights = np.array(weights)
        # internals are stored in a file in instance-level
        np.savetxt(library_path + dataset_name + '/' + library_name + '_instance_internals.txt', weights, fmt='%.8f',
                   delimiter=',')
        t2 = time.clock()
        print '[Internal Weights] finish calculation, time consumption:' + str(t2 - t1) + 's'

    # convert instance-level internals into cluster-level internals in the form of dictionary
    ret_weights = []
    for clustering_weights, label in zip(weights, labels):
        cluster_labels = np.unique(label)
        cluster_internals = {}
        for cluster_id in cluster_labels:
            cluster_internals[cluster_id] = np.mean(clustering_weights[label == cluster_id])
        # normalize cluster-level internals to [0, 1] for each base clustering
        if normalize:
            cluster_internals = _zero_one_normalizer_for_dict(cluster_internals)
        ret_weights.append(cluster_internals)

    return ret_weights

