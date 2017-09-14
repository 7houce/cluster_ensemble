from sklearn import metrics
import numpy as np
import time

"""
pre-defined internal weight types
"""


def _silhouette(data, label):
    return metrics.silhouette_score(data, label, metric='euclidean')


avail_weight_types = {'silhouette': _silhouette}


def cal_internal_weights_for_library(library_name, savename, data,
                                     library_path='Results/', savepath='', weight_type='silhouette'):
    """
    calculate internal evaluation metrics as weights for cluster ensemble to a file

    :param library_name:
    :param savename:
    :param data:
    :param library_path:
    :param savepath:
    :param weight_type:
    :return:
    """
    weights = cal_internal_weights_for_library_as_array(data, library_name,
                                                        library_path=library_path, weight_type=weight_type)
    np.savetxt(savepath + savename, weights, fmt='%.8f', delimiter=',')
    return


def cal_internal_weights_for_library_as_array(data, library_name,
                                              library_path='Results/', weight_type='silhouette'):
    """
    calculate internal evaluation metrics as weights for cluster ensemble
    return as an numpy array

    :param data:
    :param library_name:
    :param library_path:
    :param weight_type:
    :return:
    """
    t1 = time.clock()
    labels = np.loadtxt(library_path + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    weights = []
    for label in labels:
        weights.append(avail_weight_types[weight_type](data, label))
    weights = np.array(weights)
    t2 = time.clock()
    print '[Internal Weights] finish calculation, time consumption:' + str(t2 - t1) + 's'
    return weights
