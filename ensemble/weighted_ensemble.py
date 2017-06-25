from utils import io_func
import evaluation.Metrics as Metrics
from ensemble import Cluster_Ensembles as ce
from ensemble import spectral_ensemble as spec
import numpy as np
from sklearn import preprocessing

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.weighted_spectral_ensemble}
_default_ensemble_method = ['CSPA', 'Spectral']


def _activate_func(value):
    """
    Activate Function (Sigmoid)
    (internal use only)
    """
    return 1/(1+np.exp(-(np.tan(value*np.pi-(np.pi/2)))))


def _activate_func_2(value):
    """
    Activate Function (Sigmoid)
    (internal use only)
    """
    _LOWER_BOUND = -10
    _UPPER_BOUND = 10
    adjusted_value = _LOWER_BOUND + (value * (_UPPER_BOUND - _LOWER_BOUND))
    return 1/(1+np.exp(adjusted_value))


def _get_weights(labels, mlset, nlset, cons_type, scale, weights_type=1):
    """
    Get weights according to the type given.
    (internal use only)
    """
    con_per_cluster = []
    con_clustering = []

    if weights_type == 1:
        """
        weights_type 1
        naive weight solution : use 0-1 consistency for both clustering and cluster level directly
        """
        for label in labels:
            con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
        for label in labels:
            con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        if scale:
            scaler = preprocessing.MinMaxScaler()
            con_clustering = scaler.fit_transform(np.array(con_clustering))

    elif weights_type == 2:
        """
        weights_type 2
        Use the 'fulfillment ratio' of expected consistency as clustering-level weights
        """
        k_values = []
        expected_cons = {}
        for label in labels:
            con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
        for label in labels:
            con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
            k_values.append(len(np.unique(label)))
        k_values = np.array(k_values, dtype=int)
        possible_k = np.unique(k_values)
        cons = np.array(con_clustering)
        for k in possible_k:
            mean_value = np.mean(cons[k_values == k])
            expected_cons[k] = mean_value
        for i in range(0, labels.shape[0]):
            con_clustering[i] /= expected_cons[k_values[i]]
        if scale:
            scaler = preprocessing.MinMaxScaler()
            con_clustering = scaler.fit_transform(np.array(con_clustering))

    elif weights_type == 3:
        """
        weights_type 3:
        Apply activate function on clustering-level consistency
        """
        for label in labels:
            con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
        for label in labels:
            con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

        if scale:
            scaler = preprocessing.MinMaxScaler()
            con_clustering = scaler.fit_transform(np.array(con_clustering))
            for i in range(0, len(con_clustering)):
                con_clustering[i] = _activate_func(con_clustering[i])
        else:
            for i in range(0, len(con_clustering)):
                con_clustering[i] = _activate_func(con_clustering[i])

    elif weights_type == 4:
        """
        weights_type 4:
        Apply activate function on cluster-level consistency
        """
        for label in labels:
            d = Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type)
            for k in d:
                d[k] = _activate_func(d[k])
            con_per_cluster.append(d)
        for label in labels:
            con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

        if scale:
            scaler = preprocessing.MinMaxScaler()
            con_clustering = scaler.fit_transform(np.array(con_clustering))

    return con_clustering, con_per_cluster


def do_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------Weighted Ensemble for library:'+str(library_name)+'-----------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_2nd_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                         constraint_file, logger, alphas, cons_type='both',
                                         ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------New Weighted Ensemble for library:'+str(library_name)+'-------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    k_values = []
    expected_cons = {}

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        k_values.append(len(np.unique(label)))
    k_values = np.array(k_values, dtype=int)
    possible_k = np.unique(k_values)
    cons = np.array(con_clustering)
    for k in possible_k:
        mean_value = np.mean(cons[k_values == k])
        expected_cons[k] = mean_value
    for i in range(0, labels.shape[0]):
        con_clustering[i] /= expected_cons[k_values[i]]
    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_3rd_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------Weighted Ensemble for library:'+str(library_name)+'-----------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func(con_clustering[i])
    else:
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func(con_clustering[i])

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_4th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------Weighted Ensemble for library:'+str(library_name)+'-----------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        d = Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type)
        for k in d:
            d[k] = _activate_func(d[k])
        con_per_cluster.append(d)
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))


    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_5th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------Weighted Ensemble for library:'+str(library_name)+'-----------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func_2(con_clustering[i])
    else:
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func_2(con_clustering[i])

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_6th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------Weighted Ensemble for library:'+str(library_name)+'-----------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        d = Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type)
        for k in d:
            d[k] = _activate_func_2(d[k])
        con_per_cluster.append(d)
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))


    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis


def do_7th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                         constraint_file, logger, alphas, internals, cons_type='both',
                                         ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    logger.debug('===========================================================================================')
    logger.debug('-----------------New Weighted Ensemble for library:'+str(library_name)+'-------------------')
    logger.debug('-----------------Weight type = ' + cons_type + '-------------------------------------------')
    logger.debug('-----------------Scale type = ' + str(scale) + '-------------------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    k_values = []
    expected_cons = {}

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        k_values.append(len(np.unique(label)))
    k_values = np.array(k_values, dtype=int)
    possible_k = np.unique(k_values)
    cons = np.array(con_clustering)
    for k in possible_k:
        mean_value = np.mean(cons[k_values == k])
        expected_cons[k] = mean_value
    for i in range(0, labels.shape[0]):
        con_clustering[i] /= expected_cons[k_values[i]] * internals[i]
    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<-------------------------------')
    logger.debug('===========================================================================================')
    return nmis

