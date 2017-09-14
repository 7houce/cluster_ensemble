from __future__ import print_function
import numpy as np
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as Metrics
from utils import io_func
from sklearn import preprocessing
import ensemble.spectral_ensemble as spec

_default_ensemble_methods = ['CSPA', 'Spectral']

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.spectral_ensemble}


def k_selection_ensemble(labels, k_threshold, logger, weighted=False,
                         alpha=0, mlset=None, nlset=None, ctype='both'):
    """
    do selection ensemble using k as criteria
    clusteing with k smaller than k_threshold will be removed

    :param labels:
    :param k_threshold:
    :param logger:
    :param weighted: weighted version or not
    :param alpha: balance factor that control the importance of clustering/cluster
                  consistency in weights (weighted version only)
    :param mlset: cannot-link set (weighted version only)
    :param nlset: must-link set (weighted version only)
    :param ctype: type of consistency (weighted version only)
    :return:
    """
    k_value = []
    class_num = len(np.unique(labels[-1]))
    # select those clusterings that k larger than the threshold.
    for label in labels[0:-5]:
        k_value.append(len(np.unique(label)))
    k_value = np.array(k_value)
    idx = k_value.ravel() >= k_threshold
    selected_labels = labels[0:-5][idx]

    # weights
    con_per_cluster = []
    con_clustering = []
    if weighted:
        for label in selected_labels:
            con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=ctype))
        for label in selected_labels:
            con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=ctype))

    logger.debug('[K] Start consensus...shape='+str(selected_labels.shape))
    logger.debug('[K] Average k is ' + str(np.mean(k_value[idx])))
    if weighted:
        logger.debug('[K] weighted consensus, alpha='+str(alpha))

    label_CSPA = ce.cluster_ensembles_CSPAONLY(selected_labels, N_clusters_max=class_num, weighted=weighted,
                                               clustering_weights=con_clustering, cluster_level_weights=con_per_cluster,
                                               alpha=alpha)
    label_HGPA = ce.cluster_ensembles_HGPAONLY(selected_labels, N_clusters_max=class_num, weighted=weighted,
                                               clustering_weights=con_clustering, cluster_level_weights=con_per_cluster,
                                               alpha=alpha)
    label_MCLA = ce.cluster_ensembles_MCLAONLY(selected_labels, N_clusters_max=class_num, weighted=weighted,
                                               clustering_weights=con_clustering, cluster_level_weights=con_per_cluster,
                                               alpha=alpha)

    nmi_CSPA = Metrics.normalized_max_mutual_info_score(label_CSPA, labels[-1])
    nmi_HGPA = Metrics.normalized_max_mutual_info_score(label_HGPA, labels[-1])
    nmi_MCLA = Metrics.normalized_max_mutual_info_score(label_MCLA, labels[-1])
    logger.debug('CSPA performance:'+str(nmi_CSPA))
    logger.debug('HGPA performance:'+str(nmi_HGPA))
    logger.debug('MCLA performance:'+str(nmi_MCLA))
    logger.debug('--------------------------------------------')
    return


def k_selection_ensemble_for_library(library_folder, library_name, k_threshold, logger, weighted=False, alpha=0,
                                     mlset=None, nlset=None, ctype='both'):
    """
    do selection ensemble using k as criteria
    wrapper function

    :param library_folder:
    :param library_name:
    :param k_threshold:
    :param logger:
    :param weighted:
    :param alpha:
    :param mlset:
    :param nlset:
    :param ctype:
    :return:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    k_selection_ensemble(labels, k_threshold, logger, weighted=weighted, alpha=alpha, mlset=mlset, nlset=nlset,
                         ctype=ctype)
    return


def consistency_selection_ensemble(labels, mlset, nlset, logger, must_threshold, cannot_threshold, normalized=True,
                                   weighted=False, weighted_type='both', alpha=1):
    """
    do selection ensemble using must/cannot consistency as criteria
    clusteing with k smaller than k_threshold will be removed

    :param labels:
    :param mlset:
    :param nlset:
    :param logger:
    :param must_threshold:
    :param cannot_threshold:
    :param normalized:
    :param weighted:
    :param weighted_type:
    :param alpha:
    :return:
    """
    class_num = len(np.unique(labels[-1]))
    must_consistencies = []
    cannot_consistencies = []
    clustering_weights = []
    cluster_level_weights = []
    k_value = []
    for label in labels[0:-5]:
        must_cons = Metrics.consistency(label, mlset, nlset, cons_type='must')
        cannot_cons = Metrics.consistency(label, mlset, nlset, cons_type='cannot')
        if weighted:
            clustering_weights.append(Metrics.consistency(label, mlset, nlset, cons_type=weighted_type))
            cluster_level_weights.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=weighted_type))
        must_consistencies.append(must_cons)
        cannot_consistencies.append(cannot_cons)
        k_value.append(len(np.unique(label)))
    if normalized:
        scaler = preprocessing.MinMaxScaler()
        must_consistencies = scaler.fit_transform(np.array(must_consistencies).reshape(-1, 1)).ravel()
        cannot_consistencies = scaler.fit_transform(np.array(cannot_consistencies).reshape(-1, 1)).ravel()
    idx = np.logical_and(must_consistencies >= must_threshold, cannot_consistencies >= cannot_threshold)
    selected_labels = labels[0:-5][idx]
    k_value = np.array(k_value)[idx]
    logger.debug('[Consistency] Start consensus...shape=' + str(selected_labels.shape))
    if selected_labels.shape[0] == 0:
        logger.debug('[Consistency] No clusterings are selected. Out.')
        return
    logger.debug('[Consistency] Average k is '+str(np.mean(k_value)))
    label_CSPA = ce.cluster_ensembles_CSPAONLY(selected_labels, N_clusters_max=class_num, weighted=weighted,
                                               clustering_weights=clustering_weights,
                                               cluster_level_weights=cluster_level_weights, alpha=alpha)
    label_HGPA = ce.cluster_ensembles_HGPAONLY(selected_labels, N_clusters_max=class_num, weighted=weighted,
                                               clustering_weights=clustering_weights,
                                               cluster_level_weights=cluster_level_weights, alpha=alpha)
    label_MCLA = ce.cluster_ensembles_MCLAONLY(selected_labels, N_clusters_max=class_num)
    nmi_CSPA = Metrics.normalized_max_mutual_info_score(label_CSPA, labels[-1])
    nmi_HGPA = Metrics.normalized_max_mutual_info_score(label_HGPA, labels[-1])
    nmi_MCLA = Metrics.normalized_max_mutual_info_score(label_MCLA, labels[-1])
    logger.debug('CSPA performance:'+str(nmi_CSPA))
    logger.debug('HGPA performance:'+str(nmi_HGPA))
    logger.debug('MCLA performance:'+str(nmi_MCLA))
    return


def consistency_selection_ensemble_for_library(library_folder, library_name, mlset, nlset, logger, must_threshold,
                                               cannot_threshold, normalized=True, weighted=False, weighted_type='both',
                                               alpha=1):
    """
    do selection ensemble using must/cannot consistency as criteria
    wrapper function

    :param library_folder:
    :param library_name:
    :param mlset:
    :param nlset:
    :param logger:
    :param must_threshold:
    :param cannot_threshold:
    :param normalized:
    :param weighted:
    :param weighted_type:
    :param alpha:
    :return:
    """
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    consistency_selection_ensemble(labels, mlset, nlset, logger, must_threshold, cannot_threshold,
                                   normalized=normalized, weighted=weighted, weighted_type=weighted_type, alpha=alpha)
    return


def batch_do_consistency_selection_for_library(library_folder, library_name, constraint_file, logger,
                                               threshold_tuples, normalized=True, weighted=False,
                                               weighted_type='both', alpha=1):
    logger.debug('===========================================================================================')
    logger.debug('-----------------Consistency Selection Ensemble:'+str(library_name)+'----------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')
    mlset, nlset = io_func.read_constraints(constraint_file)
    for c_tuple in threshold_tuples:
        logger.debug('------->>>>Results for Must>' + str(c_tuple[0]) + ',Cannot>'+str(c_tuple[1])+' <<<<-------')
        consistency_selection_ensemble_for_library(library_folder, library_name, mlset, nlset,
                                                   logger, c_tuple[0], c_tuple[1], normalized=normalized,
                                                   weighted=weighted, weighted_type=weighted_type,
                                                   alpha=alpha)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return


def _expected_consistency_selection(labels, mlset, nlset, cons_type='', ease_factor=1):
    n_solutions = labels.shape[0]
    k_values = []
    cons = []
    final_idx = np.array([False] * n_solutions)
    for label in labels:
        cons.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        k_values.append(len(np.unique(label)))
    cons = np.array(cons)
    k_values = np.array(k_values, dtype=int)
    possible_k = np.unique(k_values)
    for k in possible_k:
        mean_value = np.mean(cons[k_values == k])
        idx = np.logical_and(cons >= mean_value * ease_factor, k_values == k)
        final_idx = np.logical_or(final_idx, idx)
    return labels[final_idx]


def expected_consistency_selection_ensemble(labels, class_num, target, mlset, nlset, cons_type='must',
                                            ensemble_methods=_default_ensemble_methods, ease_factor=1):
    selected_labels = _expected_consistency_selection(labels, mlset, nlset, cons_type=cons_type, ease_factor=ease_factor)
    retVals = []
    retVals.append(ease_factor)
    retVals.append(selected_labels.shape[0])
    print('[INFO] Selected Solutions:'+str(selected_labels.shape[0]))
    for method in ensemble_methods:
        ensemble_labels = _ensemble_method[method](selected_labels, N_clusters_max=class_num)
        ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
        retVals.append(ensemble_nmi)
        print('[INFO] Ensemble Method:'+method)
        print('[INFO] Performance:'+str(ensemble_nmi))
    return retVals
