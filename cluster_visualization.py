from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import matplotlib.colors as colors2
import matplotlib.cm as cmx
import os
import Metrics

_colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod',
           'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']


def spin_sts(D):
    """
    Side-to-side transformation to make a ordered distance matrix

    This method is introduced in
    D. Tsafrir et al.
    Sorting points into neighborhoods (SPIN) ... , Bioinformatics Vol.21 2005

    Parameters
    ----------
    :param D: distance matrix in form of [n_obj, n_obj] ndarray

    Returns
    -------
    :return: new permutation of objects
    """
    n = D.shape[0]
    w = np.array(range(1, n + 1), dtype=float)
    w = w.T - float(n)/2
    w = np.reshape(w, (n, 1))
    P_prev = np.eye(n)
    P_prev = P_prev.astype(float)
    stop = False
    iter = 0
    while not stop:
        iter += 1
        s = np.dot(D, P_prev.T)
        s = np.dot(s, w)
        p = np.argsort(-s, axis=0)

        P = coo_matrix((np.array([1]*n), (np.squeeze(np.array(range(0, n))), np.squeeze(p))))
        P = P.toarray()

        cur_norm_term = np.dot(P, s)
        prev_norm_term = np.dot(P_prev, s)
        norm = LA.norm(cur_norm_term - prev_norm_term, np.inf)
        if norm < 1e-7:
            stop = True
        P_prev = P
        i, j = P.nonzero()
    return i, j


def permute_matrix(D, index):
    """
    re-permute the distance matrix with given permutation

    Parameters
    ----------
    :param D distance matrix to re-permute
    :param index new permutation of objects

    Returns
    -------
    :return permuted matrix
    """
    n = index.shape[0]
    total = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            row.append(D[index[i]][index[j]])
        total.append(row)
    return np.array(total)


def draw_ordered_distance_matrix(distance_matrix, origin_savepath, after_savepath):
    """
    draw orderred distance matrix in order to visualize the relationship between different solutions

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
    i, j = spin_sts(distance_matrix)
    real_pos = int(np.where(j == n_solutions - 1)[0])
    CSPA_pos = int(np.where(j == n_solutions - 4)[0])
    HGPA_pos = int(np.where(j == n_solutions - 3)[0])
    MCLA_pos = int(np.where(j == n_solutions - 2)[0])

    new_matrix = permute_matrix(distance_matrix, j)
    plt.imshow(new_matrix, vmin=np.min(new_matrix), vmax=np.max(new_matrix))
    plt.annotate('Real', xy=(real_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 20),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('CSPA', xy=(CSPA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 30),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('HGPA', xy=(HGPA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 40),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate('MCLA', xy=(MCLA_pos, n_solutions - 1), xytext=(n_solutions + 5, n_solutions - 50),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.savefig(after_savepath, format='png', dpi=240)
    return


def draw_ordered_distance_matrix_in_folder(path, suffix='nid', subfolder='odm'):
    """
    draw orderred distance matrix in order to visualize the relationship between different solutions

    Parameters
    ----------
    :param path: path that stored
    :param subfolder: sub-folder to store plots
    :param suffix: suffix

    """
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return
    if not os.path.isdir(path + subfolder + '/'):
        os.mkdir(path + subfolder + '/')
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        if suffix not in f:
            continue
        fullpath = os.path.join(path, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            distance_matrix = np.loadtxt(fullpath, delimiter=',')
            origin_savepath = path + subfolder + '/' + fname[0] + '_original.png'
            after_savepath = path + subfolder + '/' + fname[0] + '_after.png'
            draw_ordered_distance_matrix(distance_matrix, origin_savepath, after_savepath)
    return


def plot_k_distribution(labels, pos, savepath):
    """

    :param labels:
    :param pos:
    :param savepath:
    :return:
    """
    plt.clf()
    k_value = []
    for label in labels[0:-4]:
        cons = len(np.unique(label))
        k_value.append(cons)
    print (k_value)
    cm = plt.get_cmap('CMRmap')
    cNorm = colors2.Normalize(vmin=min(k_value), vmax=max(k_value))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    count = 0
    for label in labels[0:-4]:
        plt.scatter(pos[count][0], pos[count][1], c=scalarMap.to_rgba(k_value[count]))
        count += 1
    plt.title('Max k = ' + str(max(k_value)) + ' ,Min k = ' + str(min(k_value))+' ,Real k = ' + str(len(np.unique(labels[-1]))))
    plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='blue', marker='D', label='Consensus')
    plt.scatter(pos[-1:, 0], pos[-1:, 1], c='red', marker='D', label='Real')
    plt.savefig(savepath, format='png', dpi=240)

    return


def plot_k_distribution_in_file(resfile, posfile, savepath):
    """

    :param resfile:
    :param posfile:
    :param savepath:
    :return:
    """
    labels = np.loadtxt(resfile, delimiter=',')
    labels = labels.astype(int)
    pos = np.loadtxt(posfile, delimiter=',')
    plot_k_distribution(labels, pos, savepath)
    return


def plot_consistency(labels, pos, mlset, nlset, savepath, consistency_type='both'):
    """

    :param resfile:
    :param posfile:
    :param mlset:
    :param nlset:
    :param savepath:
    :param consistency_type:
    :return:
    """
    plt.clf()
    consistencies = []
    for label in labels[0:-4]:
        cons = Metrics.consistency(label, mlset, nlset, cons_type=consistency_type)
        consistencies.append(cons)
    print (consistencies)
    cm = plt.get_cmap('CMRmap')
    cNorm = colors2.Normalize(vmin=min(consistencies), vmax=max(consistencies))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    count = 0
    for label in labels[0:-4]:
        plt.scatter(pos[count][0], pos[count][1], c=scalarMap.to_rgba(consistencies[count]))
        count += 1
    plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='blue', marker='D', label='Consensus')
    plt.scatter(pos[-1:, 0], pos[-1:, 1], c='red', marker='D', label='Real')
    plt.title(consistency_type + ' Consistency , max val='+str(max(consistencies))+' min val='+str(min(consistencies)))
    plt.savefig(savepath, format='png', dpi=240)

    return


def plot_consistency_in_file(resfile, posfile, mlset, nlset, savepath, consistency_type='both'):
    """

    :param resfile:
    :param posfile:
    :param mlset:
    :param nlset:
    :param savepath:
    :return:
    """
    labels = np.loadtxt(resfile, delimiter=',')
    labels = labels.astype(int)
    pos = np.loadtxt(posfile, delimiter=',')
    plot_consistency(labels, pos, mlset, nlset, savepath, consistency_type=consistency_type)
    return
