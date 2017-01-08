from sklearn import metrics
import numpy as np
import time

def diversityBtw2Cluster(label1,label2):
    """
    calculate the diversity between two clusterings c1 & c2
    diversity is defined as 1 - NMI(c1,c2)
    :param label1: labels of clustering 1 (as a 1-dimensional ndarray)
    :param label2: labels of clustering 2 (as a 1-dimensional ndarray)
    :return: diversity between these two clusterings as a float value
    """
    return 1 - metrics.normalized_mutual_info_score(label1, label2)


def diversityMatrix(labels):
    """
    generate the diversity matrix between given clusterings
    :param labels: labels of all given clusterings as a [n_of_clusterings * n_of_samples] ndarray
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
    :param label: labels of the clustering to calculate
    :param solutionset: labels of all clusterings
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
    :param label: label of the clustering to calculate
    :param mlset: Must-Link set
    :param nlset: can-Not-Link set
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
