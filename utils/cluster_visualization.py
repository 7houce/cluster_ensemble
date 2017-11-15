"""
Visualization Module
Author: Zhijie Lin
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import matplotlib.colors as colors2
import matplotlib.cm as cmx
import os
import evaluation.Metrics as Metrics
from sklearn import preprocessing

_colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod',
           'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']

_ADDITIONAL_RANGE = 5
_ADDITIONAL_NAMES = ['Single-KMeans', 'CSPA', 'HGPA', 'MCLA', 'Real']
_ADDITIONAL_LABELS = ['Single-KMeans', 'Consensus', 'Consensus', 'Consensus', 'Real']
_ADDITIONAL_COLORS = ['green', 'blue', 'blue', 'blue', 'red']
_ADDITIONAL_MARKERS = ['s', 'D', 'D', 'D', 'D']
_SCATTER_FONT_SIZE = 7
_ROUND_AMOUNT = 4
_LEGEND_FONT_SIZE = 7


def _round_digits(x):
    return round(x, _ROUND_AMOUNT)


def _spin_sts(distance_matrix, threshold=1e-12):
    """
    Side-to-side transformation to make a ordered distance matrix (internal use only)
    This method is introduced in
    D. Tsafrir et al.  Sorting points into neighborhoods (SPIN) ... , Bioinformatics Vol.21 2005
    """
    n = distance_matrix.shape[0]
    w = np.array(range(1, n + 1), dtype=float)
    w = w.T - float(n)/2
    w = np.reshape(w, (n, 1))
    P_prev = np.eye(n)
    P_prev = P_prev.astype(float)
    stop = False
    iter = 0
    while not stop:
        iter += 1
        s = np.dot(distance_matrix, P_prev.T)
        s = np.dot(s, w)
        p = np.argsort(-s, axis=0)

        P = coo_matrix((np.array([1]*n), (np.squeeze(np.array(range(0, n))), np.squeeze(p))))
        P = P.toarray()

        cur_norm_term = np.dot(P, s)
        prev_norm_term = np.dot(P_prev, s)
        norm = LA.norm(cur_norm_term - prev_norm_term, np.inf)
        if norm < threshold:
            stop = True
        P_prev = P
        i, j = P.nonzero()
    return i, j


def _permute_matrix(D, index):
    """
    re-permute the distance matrix with given permutation (internal use only)
    """
    n = index.shape[0]
    total = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            row.append(D[index[i]][index[j]])
        total.append(row)
    return np.array(total)


def plot_ordered_distance_matrix(distance_matrix, origin_savepath, after_savepath):
    """
    plot ordered distance matrix in order to visualize the relationship between different solutions

    Parameters
    ----------
    :param distance_matrix: distance matrix
    :param origin_savepath: location to store visualized original distance matrix
    :param after_savepath: location to store visualized ordered distance matrix

    """
    plt.clf()
    n_solutions = distance_matrix.shape[0]
    plt.imshow(distance_matrix, vmin=np.min(distance_matrix), vmax=np.max(distance_matrix))
    plt.savefig(origin_savepath, format='png', dpi=240)
    plt.clf()
    i, j = _spin_sts(distance_matrix)
    real_pos = int(np.where(j == n_solutions - 1)[0])
    CSPA_pos = int(np.where(j == n_solutions - 4)[0])
    HGPA_pos = int(np.where(j == n_solutions - 3)[0])
    MCLA_pos = int(np.where(j == n_solutions - 2)[0])
    single_km_pos = int(np.where(j == n_solutions - 5)[0])

    new_matrix = _permute_matrix(distance_matrix, j)
    plt.imshow(new_matrix, vmin=np.min(new_matrix), vmax=np.max(new_matrix))
    plt.annotate('Real', xy=(real_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 20),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('CSPA', xy=(CSPA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 30),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('HGPA', xy=(HGPA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 40),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('MCLA', xy=(MCLA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 50),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('KM', xy=(single_km_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 60),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.savefig(after_savepath, format='png', dpi=240)
    return


def plot_ordered_distance_matrix_for_library(library_folder, library_name, dis_suffix='nid'):
    """
    plot odm for an existing library(wrapper function)

    Parameters
    ----------
    :param library_folder:
    :param library_name:
    :param dis_suffix:
    """
    distance_matrix = np.loadtxt(library_folder + library_name + '_' + dis_suffix + '.txt', delimiter=',')
    plot_ordered_distance_matrix(distance_matrix, library_folder + library_name + '_original_distance.png',
                                 library_folder + library_name + '_odm.png')
    return


def _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, legend_need=True,
                              xlabel=None, ylabel=None, title=None):
    """
    generalized function to plot scatter graph (internal use only)
    """
    plt.clf()
    point_count = pos.shape[0]
    for i in range(0, point_count):
        plt.scatter(pos[i][0], pos[i][1], c=colors[i], marker=markers[i], label=plot_labels[i])
        plt.text(pos[i][0], pos[i][1], str(texts[i]), fontsize=_SCATTER_FONT_SIZE)
    if legend_need:
        plt.legend(loc='best', fontsize=_LEGEND_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.savefig(savepath, format='png', dpi=240)
    return


def plot_k_distribution(labels, pos, savepath):
    """
    plot k distribution of given library

    Parameters
    ----------
    :param labels:
    :param pos:
    :param savepath:
    """
    texts = []
    colors = []
    plot_labels = [None] * (len(labels) - _ADDITIONAL_RANGE)
    markers = ['o'] * (len(labels) - _ADDITIONAL_RANGE)
    for label in labels[0:-_ADDITIONAL_RANGE]:
        cons = len(np.unique(label))
        texts.append(cons)
    cNorm = colors2.Normalize(vmin=min(texts), vmax=max(texts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('CMRmap'))
    plot_labels.extend(_ADDITIONAL_NAMES)
    for text in texts:
        colors.append(scalarMap.to_rgba(text))
    texts.append('')
    texts.extend(_ADDITIONAL_NAMES[1:])
    colors.extend(_ADDITIONAL_COLORS)
    markers.extend(_ADDITIONAL_MARKERS)
    title = 'Max k = ' + str(max(texts[0:-_ADDITIONAL_RANGE])) +\
            ' ,Min k = ' + str(min(texts[0:-_ADDITIONAL_RANGE])) +\
            ' ,Real k = ' + str(len(np.unique(labels[-1])))
    _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, title=title)
    return


def plot_k_distribution_for_library(library_folder, library_name, pos_suffix='mds2d'):
    """
    plot k distribution of an existing library(wrapper function)

    Parameters
    ----------
    :param library_folder:
    :param library_name:
    :param pos_suffix:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    pos = np.loadtxt(library_folder + library_name + '_' + pos_suffix + '.txt', delimiter=',')
    plot_k_distribution(labels, pos, library_folder + library_name + '_k_distribution.png')
    return


def plot_consistency(labels, pos, mlset, nlset, savepath, consistency_type='both'):
    """
    plot consistency distribution of given library

    Parameters
    ----------
    :param labels:
    :param pos:
    :param mlset:
    :param nlset:
    :param savepath:
    :param consistency_type:
    """
    texts = []
    colors = []
    plot_labels = [None] * (len(labels) - _ADDITIONAL_RANGE)
    markers = ['o'] * (len(labels) - _ADDITIONAL_RANGE)
    for label in labels[0:-_ADDITIONAL_RANGE]:
        cons = Metrics.consistency(label, mlset, nlset, cons_type=consistency_type)
        texts.append(cons)
    cNorm = colors2.Normalize(vmin=min(texts), vmax=max(texts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('CMRmap'))
    plot_labels.extend(_ADDITIONAL_NAMES)
    for text in texts:
        colors.append(scalarMap.to_rgba(text))
    texts = map(_round_digits, texts)
    texts.append('')
    texts.extend(_ADDITIONAL_NAMES[1:])
    colors.extend(_ADDITIONAL_COLORS)
    markers.extend(_ADDITIONAL_MARKERS)
    title = consistency_type + ' Consistency ,' + 'Max val = ' + str(max(texts[0:-_ADDITIONAL_RANGE])) +\
                               ' ,Min k = ' + str(min(texts[0:-_ADDITIONAL_RANGE]))
    _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, title=title)
    return


def plot_consistency_for_library(library_folder, library_name, mlset, nlset, pos_suffix='mds2d',
                                 consistency_type='both'):
    """
    plot consistency distribution of an existing library(wrapper function)

    Parameters
    ----------
    :param library_folder:
    :param library_name:
    :param mlset:
    :param nlset:
    :param pos_suffix:
    :param consistency_type:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    pos = np.loadtxt(library_folder + library_name + '_' + pos_suffix + '.txt', delimiter=',')
    plot_consistency(labels, pos, mlset, nlset, library_folder + library_name + '_consistency_'+consistency_type+'.png',
                     consistency_type=consistency_type)
    return


def plot_nmi_max(labels, pos, savepath):
    """
    plot nmi_max distribution of given library

    Parameters
    ----------
    :param labels:
    :param pos:
    :param savepath:
    """
    texts = []
    colors = []
    plot_labels = [None] * (len(labels) - _ADDITIONAL_RANGE)
    markers = ['o'] * (len(labels) - _ADDITIONAL_RANGE)
    for label in labels[0:-1]:
        cons = Metrics.normalized_max_mutual_info_score(label, labels[-1])
        texts.append(cons)
    cNorm = colors2.Normalize(vmin=min(texts[0:-_ADDITIONAL_RANGE+1]), vmax=max(texts[0:-_ADDITIONAL_RANGE+1]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('CMRmap'))
    plot_labels.extend(_ADDITIONAL_NAMES)
    for text in texts[0:-_ADDITIONAL_RANGE+1]:
        colors.append(scalarMap.to_rgba(text))
    texts = map(_round_digits, texts)
    texts.extend(_ADDITIONAL_NAMES[-1:])
    colors.extend(_ADDITIONAL_COLORS)
    markers.extend(_ADDITIONAL_MARKERS)
    title = 'NMI distribution, ' + 'Max val = ' + str(max(texts[0:-_ADDITIONAL_RANGE])) +\
            ' ,Min k = ' + str(min(texts[0:-_ADDITIONAL_RANGE]))
    _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, title=title)
    return


def plot_nmimax_for_library(library_folder, library_name, pos_suffix='mds2d'):
    """
    plot nmi_max distribution of an existing library(wrapper function)

    Parameters
    ----------
    :param library_folder:
    :param library_name:
    :param pos_suffix:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    pos = np.loadtxt(library_folder + library_name + '_' + pos_suffix + '.txt', delimiter=',')
    plot_nmi_max(labels, pos, library_folder + library_name + '_nmimax_distribution.png')
    return


def plot_mst_result(mstmodel, pos, savepath):
    """
    plot mst result of given mst model
    TODO: refactor

    Parameters
    ----------
    :param mstmodel:
    :param pos:
    :param savepath:
    """
    clusters = np.unique(mstmodel.labels_)
    fig = plt.figure(1)
    plt.clf()
    for i in clusters:
        xs = pos[0:-5][mstmodel.labels_ == i, 0]
        ys = pos[0:-5][mstmodel.labels_ == i, 1]
        ax = plt.axes([0., 0., 1., 1.])
        if i != -1:
            plt.scatter(xs, ys, c=_colors[((int(i) + 1) % len(_colors))], label='Clusters-' + str(i))
        else:
            plt.scatter(xs, ys, c=_colors[((int(i) + 1) % len(_colors))], label='Outliers')
    plt.scatter(pos[-5, 0], pos[-5, 1], c='green', marker='s', label='Single-KMeans')
    plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='blue', marker='D', label='Consensus')
    plt.scatter(pos[-1:, 0], pos[-1:, 1], c='red', marker='D', label='Real')
    plt.text(pos[-4, 0], pos[-4, 1], 'CSPA', fontsize=7)
    plt.text(pos[-3, 0], pos[-3, 1], 'HGPA', fontsize=7)
    plt.text(pos[-2, 0], pos[-2, 1], 'MCLA', fontsize=7)
    plt.legend(loc='best', shadow=True, fontsize=8)
    plt.savefig(savepath, format='png', dpi=240)
    return


def plot_normalized_consistency(labels, mlset, nlset, savepath, additional_values):
    """
    plot correlations between must and cannot consistency of given library

    Parameters
    ----------
    :param labels:
    :param mlset:
    :param nlset:
    :param savepath:
    :param additional_values:
    """
    texts = additional_values
    colors = []
    plot_labels = [None] * (len(labels) - _ADDITIONAL_RANGE)
    markers = ['o'] * (len(labels) - _ADDITIONAL_RANGE)
    cNorm = colors2.Normalize(vmin=min(texts), vmax=max(texts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('CMRmap'))
    plot_labels.extend(_ADDITIONAL_NAMES)
    for text in texts:
        colors.append(scalarMap.to_rgba(text))
    title = 'Must-Cannot Correlation'
    must_consistencies = []
    cannot_consistencies = []
    for label in labels[0:-5]:
        must_cons = Metrics.consistency(label, mlset, nlset, cons_type='must')
        cannot_cons = Metrics.consistency(label, mlset, nlset, cons_type='cannot')
        must_consistencies.append(must_cons)
        cannot_consistencies.append(cannot_cons)
    scaler = preprocessing.MinMaxScaler()
    must_consistencies = scaler.fit_transform(np.array(must_consistencies).reshape(-1, 1))
    cannot_consistencies = scaler.fit_transform(np.array(cannot_consistencies).reshape(-1, 1))
    pos = np.hstack((np.array(must_consistencies), np.array(cannot_consistencies)))
    _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, title=title,
                              xlabel='Must consistency', ylabel='Cannot consistency', legend_need=False)
    return


def plt_consistency_corelation_with_k(labels, mlset, nlset, savepath):
    """
    plot correlations between must and cannot consistency of given library with k-value(wrapper function)

    Parameters
    ----------
    :param labels:
    :param mlset:
    :param nlset:
    :param savepath:
    """
    k_value = []
    for label in labels[0:-5]:
        cons = len(np.unique(label))
        k_value.append(cons)
    print (k_value)
    plot_normalized_consistency(labels, mlset, nlset, savepath, k_value)
    return


def plot_consistency_corelation_for_solution(library_folder, library_name, mlset, nlset):
    """
    plot correlations between must and cannot consistency of an existing library with k-value(wrapper function)

    Parameters
    ----------
    :param library_folder:
    :param library_name:
    :param mlset:
    :param nlset:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    plt_consistency_corelation_with_k(labels, mlset, nlset, library_folder + library_name + '_normalized.png')
    return


def plot_for_all_library(path, suffix='res', type='k', mlset=None, nlset=None):
    """
    plot specific visualizations for all libraries in given folder

    :param path: path of the given folder
    :param suffix: suffix of result file
    :param type: type of graph to plot
    :param mlset: must-link set (for consistency only)
    :param nlset: cannot-link set (for consistency only)
    :return:
    """
    if not os.path.isdir(path):
        return
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        if suffix not in f:
            continue
        fullpath = os.path.join(path, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            if type == 'all':
                    plot_k_distribution_for_library(path, fname[0])
                    plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='both')
                    plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='must')
                    plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='cannot')
                    plot_nmimax_for_library(path, fname[0])
                    plot_consistency_corelation_for_solution(path, fname[0], mlset, nlset)
            if type == 'k':
                plot_k_distribution_for_library(path, fname[0])
            if type == 'cons_both':
                plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='both')
            if type == 'cons_must':
                plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='must')
            if type == 'cons_cannot':
                plot_consistency_for_library(path, fname[0], mlset, nlset, consistency_type='cannot')
            if type == 'nmi':
                plot_nmimax_for_library(path, fname[0])
            if type == 'n_corelation':
                plot_consistency_corelation_for_solution(path, fname[0], mlset, nlset)
            if type == 'odm':
                plot_ordered_distance_matrix_for_library(path, fname[0])
    return


def plot_k_consistency_distribution(labels, mlset, nlset, savepath, pure=True, cons_type='must'):
    k_value = []
    if not pure:
        labels = labels[0:-5]
    for label in labels:
        cons = len(np.unique(label))
        k_value.append(cons)

    texts = [''] * len(labels)
    plot_labels = [None] * len(labels)
    markers = ['x'] * len(labels)
    colors = ['blue'] * len(labels)
    title = 'k-'+cons_type+' consistency Correlation'

    consistencies = []
    for label in labels:
        cons = Metrics.consistency(label, mlset, nlset, cons_type=cons_type)
        consistencies.append(cons)
    pos = np.hstack((np.array(k_value).reshape(-1, 1), np.array(consistencies).reshape(-1, 1)))
    print (pos.shape)

    _plot_generalized_scatter(pos, colors, texts, markers, plot_labels, savepath, title=title,
                              xlabel='k', ylabel='consistency', legend_need=False)
    return

