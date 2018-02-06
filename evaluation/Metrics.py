"""
Metric Calculation for NMI, quality, diversity, consistency ....
Author: Zhijie Lin
"""
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


def _avg_consistency(label, mlset, nlset):
    mlength = float(len(mlset))
    nlength = float(len(nlset))
    mlcount = 0
    nlcount = 0
    for obj1, obj2 in mlset:
        if label[obj1] == label[obj2]:
            mlcount += 1
    for obj1, obj2 in nlset:
        if label[obj1] != label[obj2]:
            nlcount += 1
    return ((mlcount / mlength) + (nlcount / nlength)) / float(2)


def consistency(label, mlset, nlset, cons_type='both'):
    """
    calculate consistency of a given clustering under a set of Must-Link constraint and can-Not Link constraint
    we simply apply 0-1 consistency here

    Parameters
    ----------
    :param label: label of the clustering to calculate
    :param mlset: Must-Link set
    :param nlset: can-Not-Link set
    :param cons_type : type of consistency
                       'both'    : both must-link and cannot-link constraints are considered
                       'must'    : only must-link constraints are considered
                       'cannot'  : only cannot-link constraints are considered
                       'avg'     : average of must-link consistency and cannot-link consistency
    Returns
    -------
    :return: consistency as a float value
    """
    if cons_type == 'avg':
        return _avg_consistency(label, mlset, nlset)
    if cons_type == 'must':
        setlength = float(len(mlset))
    elif cons_type == 'cannot':
        setlength = float(len(nlset))
    else:
        setlength = float(len(mlset) + len(nlset))
    mlcount = 0
    nlcount = 0
    if cons_type != 'cannot':
        for obj1, obj2 in mlset:
            if label[obj1] == label[obj2]:
                mlcount += 1
    if cons_type != 'must':
        for obj1, obj2 in nlset:
            if label[obj1] != label[obj2]:
                nlcount += 1
    if setlength == 0:
        return 1.0
    else:
        return float(nlcount + mlcount) / setlength


def _extract_keys(dic1, dic2, zero=True):
    llist = []
    if zero:
        for k, v in dic2.items():
            llist.append(v)
        return llist
    else:
        for k, v in dic1.items():
            if v != 0:
                llist.append(dic2[k])
        return llist


def _get_g_gamma(cluster_instances, cluster_constraints, n_constraints, n_instances, gamma=0.5):
    expected_constraints = (float(cluster_instances) / n_instances) * n_constraints
    if cluster_constraints <= expected_constraints:
        return cluster_constraints / expected_constraints * gamma
    else:
        return gamma + (1 - gamma) * (cluster_constraints - expected_constraints) / (n_constraints - expected_constraints)


def consistency_per_cluster_efficient(label, mlset, nlset, cons_type='both'):
    n_clusters = len(np.unique(label))
    satisfy_ML = [0] * n_clusters
    satisfy_CL = [0] * n_clusters
    count_ML = [0] * n_clusters
    count_CL = [0] * n_clusters
    consistencies = {}
    cluster_constraints = {}
    for obj1, obj2 in mlset:
        if label[obj1] != label[obj2]:
            count_ML[label[obj1]] += 1
            count_ML[label[obj2]] += 1
        else:
            count_ML[label[obj1]] += 1
            satisfy_ML[label[obj1]] += 1
    for obj1, obj2 in nlset:
        if label[obj1] != label[obj2]:
            count_CL[label[obj1]] += 1
            count_CL[label[obj2]] += 1
            satisfy_CL[label[obj1]] += 1
            satisfy_CL[label[obj2]] += 1
    # print type(n_clusters)
    for i in range(0, n_clusters):
        if cons_type == 'both':
            cluster_constraints_i = count_CL[i] + count_ML[i]
            cluster_satisfy = satisfy_CL[i] + satisfy_ML[i]
            weight_i = float(cluster_satisfy) / cluster_constraints_i if cluster_constraints_i > 0 else 1
            # gamma_i = _get_g_gamma(len(label[label == i]), cluster_constraints_i, n_ml + n_cl, n_instances, gamma=gamma)
            consistencies[i] = weight_i
            cluster_constraints[i] = cluster_constraints_i
            # gammas[i] = gamma_i
        elif cons_type == 'must':
            cluster_constraints_i = count_ML[i]
            cluster_satisfy = satisfy_ML[i]
            weight_i = float(cluster_satisfy) / cluster_constraints_i if cluster_constraints_i > 0 else 1
            # gamma_i = _get_g_gamma(len(label[label == i]), cluster_constraints_i, n_ml, n_instances, gamma=gamma)
            consistencies[i] = weight_i
            cluster_constraints[i] = cluster_constraints_i
            # gammas[i] = gamma_i
    # if cons_type == 'both':
    #     values = np.array(_extract_keys(cluster_constraints, consistencies))
    #     values = values[values != 0.0]
    #     values = values[values != 1.0]
    #     lowerbound = np.percentile(values, 25)
    #     upperbound = np.percentile(values, 75)
    #     if upperbound != lowerbound:
    #         for k in consistencies:
    #             if consistencies[k] > upperbound:
    #                 consistencies[k] = 1.0
    #             elif consistencies[k] < lowerbound:
    #                 consistencies[k] = 0.0
    #             else:
    #                 consistencies[k] = (consistencies[k] - lowerbound) / (upperbound - lowerbound)

    return consistencies, cluster_constraints


def consistency_per_cluster(label, mlset, nlset, cons_type='both'):
    """
    calculate consistencies of each cluster in a given clustering
    under a set of Must-Link constraint and cannot Link constraint
    we simply apply 0-1 consistency here

    Parameters
    ----------
    :param label: label of the clustering to calculate
    :param mlset: Must-Link set
    :param nlset: can-Not-Link set
    :param cons_type : type of consistency
                       'both'    : both must-link and cannot-link constraints are considered
                       'must'    : only must-link constraints are considered
                       'cannot'  : only cannot-link constraints are considered

    Returns
    -------
    :return: consistencies of each cluster in a dictionary
    """
    consistencies = {}
    clusters = np.unique(label)
    # consistencies are calculated for each cluster in the clustering
    for cluster in clusters:
        mlength = 0
        nlength = 0
        mlcount = 0
        nlcount = 0
        for obj1, obj2 in mlset:
            # if 2 samples in the constraint do not exist in this cluster, this constraint will be ignored.
            if label[obj1] != cluster and label[obj2] != cluster:
                continue
            if label[obj1] == label[obj2]:
                mlcount += 1
                mlength += 1
            else:
                mlength += 1
        for obj1, obj2 in nlset:
            if label[obj1] != cluster and label[obj2] != cluster:
                continue
            if label[obj1] != label[obj2]:
                nlcount += 1
                nlength += 1
            else:
                nlength += 1
        if mlength == 0 and nlength == 0:
            consistencies[cluster] = 1
        elif cons_type == 'avg':
            m_c = float(mlcount) / mlength if mlength > 0 else 1
            n_c = float(nlcount) / nlength if nlength > 0 else 1
            consistencies[cluster] = (m_c + n_c) / float(2)
        elif cons_type == 'must':
            consistencies[cluster] = float(mlcount) / mlength if mlength > 0 else 1
        elif cons_type == 'cannot':
            consistencies[cluster] = float(nlcount) / nlength if nlength > 0 else 1
        else:
            consistencies[cluster] = float(mlcount + nlcount) / (mlength + nlength)
    return consistencies


def average_consistency(solution_label, labels, mlset, nlset, cons_type='both'):
    """
    compute average consistency for each cluster of solutions

    Parameters
    ----------
    :param solution_label: cluster labels of solutions
    :param labels: cluster labels of instances in all solutions (result matrix)
    :param mlset: must-link constraints
    :param nlset: cannot-link constraints
    :param cons_type : type of consistency
                       'both'    : both must-link and cannot-link constraints are considered
                       'must'    : only must-link constraints are considered
                       'cannot'  : only cannot-link constraints are considered

    Returns
    -------
    :return: dict with solution cluster labels as keys and avg_consistency as values
    """
    sol_clusters = np.unique(solution_label)
    avg_cons = {}
    for sol_cluster in sol_clusters:
        total_cons = 0.0
        label_matrix = labels[solution_label == sol_cluster]
        n_solutions = label_matrix.shape[0]
        for label in label_matrix:
            total_cons += consistency(label, mlset, nlset, cons_type)
        avg_cons[sol_cluster] = total_cons / float(n_solutions)
    return avg_cons


def find_best_match(label_true, label_pred):
    """
    find best match between true label and predicted label

    Parameters
    ----------
    :param label_true: true labels as 1d array
    :param label_pred: predicted labels as 1d array

    Returns
    -------
    :return: match as a dictionary
    """
    if len(label_true) != len(label_pred):
        raise ValueError("[FIND_BEST_MATCH] length of true labels and predicted labels should be the same")
    best_match = dict(zip(np.unique(label_pred), [-1]*len(label_true)))
    real_class = np.unique(label_true)
    predicted_cluster = np.unique(label_pred)
    match_num_matrix = []
    for clu in predicted_cluster:
        match_num = [] * len(real_class)
        for cla in real_class:
            overlap_num = np.logical_and(label_true == cla, label_pred == clu).astype(int).sum()
            match_num.append(overlap_num)
        match_num_matrix.append(match_num)
    match_num_matrix = np.array(match_num_matrix)
    for cla in real_class:
        predicted_cluster_rank = np.argsort(-match_num_matrix[:, cla])
        for clu in predicted_cluster_rank:
            if best_match[clu] == -1:
                best_match[clu] = cla
                break
    return best_match, match_num_matrix


def precision(label_true, label_pred):
    """
    clustering precision between true labels and predicted labels
    based on find_best_match

    Parameters
    ----------
    :param label_true: true labels as 1d array
    :param label_pred: predicted labels as 1d array

    Returns
    -------
    :return: precision as a double value
    """
    best_match, _ = find_best_match(label_true, label_pred)
    predicted_clusters = np.unique(label_pred)
    cur_num = 0
    n_samples = len(label_true)
    for clu in predicted_clusters:
        cur_num += np.logical_and(label_pred == clu, label_true == best_match[clu]).astype(int).sum()
    return float(cur_num) / n_samples

