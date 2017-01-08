from sklearn import metrics
from sklearn.metrics.cluster import entropy
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import time


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays (internal use only)
       Copy from sklearn.metrics.cluster.supervised since it is not defined at the '__init__'
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred



def normalized_max_mutual_info_score(labels_true, labels_pred):
    """
    A variant version of NMI that is given as:
    NMI_max = MI(U, V) / max{ H(U), H(V) }
    based on 'adjusted mutual info score' in sklearn

    Parameters
    ----------
    :param labels_true: labels of clustering 1 (as a 1-dimensional ndarray)
    :param labels_pred: labels of clustering 2 (as a 1-dimensional ndarray)
    :return: diversity between these two clusterings as a float value

    Returns
    -------
    :return: NMI-max between these two clusterings as a float value

    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = metrics.mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi_max = mi / max(h_true, h_pred)
    return nmi_max


def NIDBtw2Cluster(label1, label2):
    """
    calculate the NID (Normalized Information Distance) between 2 clusterings
    NID is given as:
    NID = 1 - NMI-max

    Parameters
    ----------
    :param label1: labels of clustering 1
    :param label2: labels of clustering 2

    Returns
    -------
    :return: NID as a float value

    """
    return 1 - normalized_max_mutual_info_score(label1, label2)


def NIDMatrix(labels):
    """
    generate the NID matrix between given clusterings

    Parameters
    ----------
    :param labels: labels of all given clusterings as a  [n_of_clusterings, n_of_samples] ndarray

    Returns
    -------
    :return: NID matrix as a [n_clusterings, n_clusterings] ndarray
    """
    nidmat = []
    num_of_clusterings = labels.shape[0]
    for i in range(0, num_of_clusterings):
        nidrow = []
        for j in range(0, num_of_clusterings):
            if i == j:
                nid = 0.0
            elif i > j:
                nid = nidmat[j][i]
            else:
                nid = NIDBtw2Cluster(labels[i], labels[j])
            nidrow.append(nid)
        nidmat.append(nidrow)
    return np.array(nidmat)


def diversityBtw2Cluster(label1,label2):
    """
    calculate the diversity between two clusterings c1 & c2
    diversity is defined as 1 - NMI(c1,c2)

    Parameters
    ----------
    :param label1: labels of clustering 1 (as a 1-dimensional ndarray)
    :param label2: labels of clustering 2 (as a 1-dimensional ndarray)

    Returns
    -------
    :return: diversity between these two clusterings as a float value
    """
    return 1 - metrics.normalized_mutual_info_score(label1, label2)


def diversityMatrix(labels):
    """
    generate the diversity matrix between given clusterings

    Parameters
    ----------
    :param labels: labels of all given clusterings as a [n_of_clusterings * n_of_samples] ndarray

    Returns
    -------
    :return: diversity matrix as a [n_of_clusterings * n_of_clusterings] ndarray
    """
    timesum = 0.0
    divMat = []
    num_of_clusterings = labels.shape[0]
    for i in range(0, num_of_clusterings):
        divRow = []
        for j in range(0, num_of_clusterings):
            time0 = time.clock()
            div = 1 - metrics.normalized_mutual_info_score(labels[i], labels[j])
            time1 = time.clock()
            timesum += (time1 - time0)
            divRow.append(div)
        divMat.append(divRow)
    print 'total cal time='+str(timesum)
    return np.array(divMat)


# TODO: need further confirmation about the implementation
def quality(label, solutionset):
    """
    calculate the quality of a clustering in a set of clusterings
    quality is defined as the average of NMIs between this clustering and all clusterings in the set

    Parameters
    ----------
    :param label: labels of the clustering to calculate
    :param solutionset: labels of all clusterings

    Returns
    -------
    :return: quality as double value
    """
    totalNMI = 0.0
    count = 0
    for solution in solutionset:
        totalNMI += metrics.normalized_mutual_info_score(label, solution)
        count += 1
    avgquality = totalNMI / count
    return avgquality


def consistency(label, mlset, nlset):
    """
    calculate consistency of a given clustering under a set of Must-Link constraint and can-Not Link constraint
    we simply apply 0-1 consistency here

    Parameters
    ----------
    :param label: label of the clustering to calculate
    :param mlset: Must-Link set
    :param nlset: can-Not-Link set

    Returns
    -------
    :return: consistency as a float value
    """
    setlength = float(len(mlset) + len(nlset))
    mlcount = 0
    nlcount = 0
    for obj1, obj2 in mlset:
        if label[obj1] == label[obj2]:
            mlcount += 1
    for obj1, obj2 in nlset:
        if label[obj1] != label[obj2]:
            nlcount += 1
    return float(nlcount + mlcount) / setlength
