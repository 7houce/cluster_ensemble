import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import os


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
    print w.shape
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
        print P
        i, j = P.nonzero()
    return i, j


def permuteMatrix(D, index):
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


def drawOrderedDistanceMatrix(path, subfolder='odm'):
    """
    draw orderred distance matrix in order to visualize the relationship between different solutions

    Parameters
    ----------
    :param path: path that stored
    :param subfolder: sub-folder to store plots

    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return
    for f in os.listdir(path):
        fullpath = os.path.join(path, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            distanceMatrix = np.loadtxt(fullpath, delimiter=',')
            plt.imshow(distanceMatrix, vmin=np.min(distanceMatrix), vmax=np.max(distanceMatrix))
            plt.savefig(path + subfolder + '/' + fname[0] + '_original.svg', format='svg', dpi=240)
            plt.clf()
            i, j = spin_sts(distanceMatrix)
            newMatrix = permuteMatrix(distanceMatrix, j)
            plt.imshow(newMatrix, vmin=np.min(newMatrix), vmax=np.max(newMatrix))
            plt.savefig(path + subfolder + '/' + fname[0] + '_after.svg', format='svg', dpi=240)
