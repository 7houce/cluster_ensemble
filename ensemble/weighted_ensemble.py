from utils import io_func
import evaluation.Metrics as Metrics
from ensemble import Cluster_Ensembles as ce
from ensemble import spectral_ensemble as spec
import numpy as np

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.weighted_spectral_ensemble}
_default_ensemble_method = ['CSPA', 'Spectral']


def do_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file, logger, alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method):
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
