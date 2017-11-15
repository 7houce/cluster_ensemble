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


def _zero_one_normalizer(weights):
    max_value = np.max(weights)
    min_value = np.min(weights)
    weights -= min_value
    weights /= (max_value - min_value)
    return weights


_avail_weight_types = {'silhouette': _silhouette}


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
