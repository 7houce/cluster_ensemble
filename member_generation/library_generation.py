"""
Generate a clustering library
Author: Zhijie Lin
"""
from __future__ import print_function
import os
import numpy as np
import random as rand
from sklearn import cluster
from sklearn import manifold
import subspace as sm
import ensemble.Cluster_Ensembles as ce
import utils.cluster_visualization as cv
import utils.io_func as io_func
import constrained_methods.efficient_cop_kmeans as eck
import constrained_methods.constrained_clustering as cc
import evaluation.Metrics as Metrics
# from ..ensemble import Cluster_Ensembles as ce
# from ..utils import cluster_visualization as cv
# from ..utils import io_func
# from ..constrained_methods import efficient_cop_kmeans as eck
# from ..constrained_methods import constrained_clustering as cc
# from ..evaluation import Metrics
_sampling_methods = {'FSRSNN': sm.FSRSNN,
                     'FSRSNC': sm.FSRSNC}

_constrained_methods = {'E2CP': cc.E2CP,
                        'Cop_KMeans': eck.cop_kmeans_wrapper}

_INT_MAX = 2147483647


def _get_file_name(dataset_name, n_cluster_lower_bound, n_cluster_upper_bound,
                   feature_sampling, feature_sampling_lower_bound,
                   sample_sampling, sample_sampling_lower_bound,
                   n_members, f_stable, s_stable, sampling_method, is_constraint_method=False, constraint_file=None):
    """
    get file name to store the matrix (internal use only)
    """
    if not f_stable and feature_sampling > 1.0:
        f_string = str(int(feature_sampling_lower_bound)) + '~' + str(int(feature_sampling))
    elif not f_stable and feature_sampling <= 1.0:
        f_string = str(feature_sampling_lower_bound) + '~' + str(feature_sampling)
    elif f_stable and feature_sampling > 1.0:
        f_string = str(int(feature_sampling))
    else:
        f_string = str(feature_sampling)

    if not s_stable and sample_sampling > 1.0:
        s_string = str(int(sample_sampling_lower_bound)) + '~' + str(int(sample_sampling))
    elif not f_stable and feature_sampling <= 1.0:
        s_string = str(sample_sampling_lower_bound) + '~' + str(sample_sampling)
    elif s_stable and sample_sampling > 1.0:
        s_string = str(int(sample_sampling))
    else:
        s_string = str(sample_sampling)
    constraint_file_suffix = '' if not is_constraint_method else ('_' + constraint_file.split('Constraints/')[1])
    filename = dataset_name + '_' + str(n_cluster_lower_bound) + '-' + str(n_cluster_upper_bound) + '_' + \
               s_string + '_' + f_string + '_' + str(n_members) + '_' + sampling_method + constraint_file_suffix
    return filename


def generate_library(data, target, dataset_name, n_members, class_num,
                     n_cluster_lower_bound=0, n_cluster_upper_bound=0,
                     feature_sampling=1.0, sample_sampling=0.7,
                     feature_sampling_lower_bound=0.05, sample_sampling_lower_bound=0.1,
                     f_stable_sample=True, s_stable_sample=True,
                     constraints_file=None, sampling_method='FSRSNC', verbose=True, path='Results/',
                     metric='nid', manifold_type='MDS', subfolder=True,
                     generate_only=False):
    """

    :param data:
    :param target:
    :param dataset_name:
    :param n_members:
    :param class_num:
    :param n_cluster_lower_bound:
    :param n_cluster_upper_bound:
    :param feature_sampling:
    :param sample_sampling:
    :param feature_sampling_lower_bound:
    :param sample_sampling_lower_bound:
    :param f_stable_sample:
    :param s_stable_sample:
    :param constraints_file:
    :param sampling_method:
    :param verbose:
    :param path:
    :param metric:
    :param manifold_type:
    :param subfolder:
    :return:
    """
    print('start generating library for dataset:' + dataset_name)

    # make sure that path to store the library existing
    if not os.path.isdir(path):
        os.mkdir(path)
    if subfolder:
        savepath = path + dataset_name + '/'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
    else:
        savepath = path

    # we set the range of cluster number to [k, 10k] if not defined
    if n_cluster_lower_bound == 0 or n_cluster_upper_bound == 0:
        n_cluster_lower_bound = class_num
        n_cluster_upper_bound = class_num * 10

    # get sampling method, if not exist, it will raise a exception
    if sampling_method in _sampling_methods.keys():
        is_constrained = False
    elif sampling_method in _constrained_methods.keys():
        is_constrained = True
    else:
        raise ValueError('ensemble generation : Method should be set properly.')

    # read constraints file if existing
    if constraints_file is not None:
        mlset, nlset = io_func.read_constraints(constraints_file)
    else:
        if is_constrained:
            raise Exception('ensemble generation : Constrained Member must be with a constraints file.')
        constraints_file = ''
        mlset = []
        nlset = []

    # lower bound of sampling rate (use only if 'stable' set to be false)
    if feature_sampling_lower_bound > feature_sampling:
        feature_sampling_lower_bound = feature_sampling / 2
    if sample_sampling_lower_bound > sample_sampling:
        sample_sampling_lower_bound = sample_sampling / 2

    # there should be at least 2 clusters in the clustering
    if n_cluster_lower_bound < 2:
        n_cluster_lower_bound = 2
    if n_cluster_upper_bound < n_cluster_lower_bound:
        n_cluster_upper_bound = n_cluster_lower_bound

    # path and filename to write the file
    filename = _get_file_name(dataset_name, n_cluster_lower_bound, n_cluster_upper_bound, feature_sampling,
                              feature_sampling_lower_bound, sample_sampling, sample_sampling_lower_bound, n_members,
                              f_stable_sample, s_stable_sample, sampling_method, is_constraint_method=is_constrained,
                              constraint_file=constraints_file)

    # we won't generate the library with same sampling rate and size if existing
    if os.path.isfile(savepath + filename + '.res'):
        print ('library already exists.')
        return
    elif os.path.isfile(savepath + filename + '_pure.res'):
        print('corresponding pure library already exists.')
        return

    tag = True

    # matrix to store clustering results
    mat = np.empty(data.shape[0])

    # generate ensemble members
    for i in range(0, n_members):
        # determine k randomly
        cluster_num = np.random.randint(n_cluster_lower_bound, n_cluster_upper_bound + 1)
        random_state = np.random.randint(0, _INT_MAX - 1)

        cur_feature_sampling = feature_sampling
        cur_sample_sampling = sample_sampling
        if not f_stable_sample:
            cur_feature_sampling = rand.uniform(feature_sampling_lower_bound, feature_sampling)
        if not s_stable_sample:
            cur_sample_sampling = rand.uniform(sample_sampling_lower_bound, sample_sampling)

        print('For this base clustering, cluster number is ' + str(cluster_num))
        # generate ensemble member by given method
        if sampling_method == 'Cop_KMeans':
            result = _constrained_methods[sampling_method](data, cluster_num, mlset, nlset)
        elif sampling_method == 'E2CP':
            e2cp = _constrained_methods[sampling_method](data=data, ml=mlset, cl=nlset, n_clusters=cluster_num)
            e2cp.fit_constrained()
            result = e2cp.labels
        else:
            result = _sampling_methods[sampling_method](data, target, r_clusters=cluster_num,
                                                        r_state=random_state, fsr=cur_feature_sampling,
                                                        ssr=cur_sample_sampling)
        # print diversity
        diver = Metrics.normalized_max_mutual_info_score(result, target)
        if verbose:
            print ('Base clustering' + str(i) + ' nmi_max between real labels = ' + str(diver))
        # stack the result into the matrix
        if tag:
            mat = np.array(result)
            mat = np.reshape(mat, (1, data.shape[0]))
            tag = False
        else:
            temp = np.array(result)
            temp = np.reshape(temp, (1, data.shape[0]))
            mat = np.vstack([mat, np.array(temp)])

    # change element type to int for consensus
    mat = mat.astype(int)

    if generate_only:
        np.savetxt(savepath + filename + '_pure' + '.res', mat, fmt='%d', delimiter=',')
        return filename+'_pure.res'

    # single k-means model, for comparison
    clf = cluster.KMeans(n_clusters=class_num)
    clf.fit(data)
    kmlabels = clf.labels_

    # do consensus
    labels_CSPA = ce.cluster_ensembles_CSPAONLY(mat, N_clusters_max=class_num)
    labels_HGPA = ce.cluster_ensembles_HGPAONLY(mat, N_clusters_max=class_num)
    labels_MCLA = ce.cluster_ensembles_MCLAONLY(mat, N_clusters_max=class_num)

    # put consensus results into the matrix
    mat = np.vstack([mat, np.reshape(kmlabels, (1, data.shape[0]))])
    mat = np.vstack([mat, np.reshape(labels_CSPA, (1, data.shape[0]))])
    mat = np.vstack([mat, np.reshape(labels_HGPA, (1, data.shape[0]))])
    mat = np.vstack([mat, np.reshape(labels_MCLA, (1, data.shape[0]))])

    # put real labels into the matrix
    temp = np.reshape(target, (1, data.shape[0]))
    mat = np.vstack([mat, np.array(temp)])

    print ('Dataset ' + dataset_name + ', consensus finished, saving...')

    # write results to external file, use %d to keep integer part only
    np.savetxt(savepath + filename + '.res', mat, fmt='%d', delimiter=',')

    # print labels and diversities (between the real labels)
    nmi_CSPA = Metrics.normalized_max_mutual_info_score(labels_CSPA, target)
    nmi_HGPA = Metrics.normalized_max_mutual_info_score(labels_HGPA, target)
    nmi_MCLA = Metrics.normalized_max_mutual_info_score(labels_MCLA, target)
    print ('consensus NMI (CSPA) =' + str(nmi_CSPA))
    print ('consensus NMI (HGPA) =' + str(nmi_HGPA))
    print ('consensus NMI (MCLA) =' + str(nmi_MCLA))

    kmnmi = Metrics.normalized_max_mutual_info_score(kmlabels, target)
    print ('single-model diversity (K-means) =' + str(kmnmi))
    # save performances
    perf = np.array([nmi_CSPA, nmi_HGPA, nmi_MCLA, kmnmi])
    np.savetxt(savepath + filename + '_performance.txt', perf, fmt='%.6f', delimiter=',')

    if is_constrained:
        return

    if metric == 'diversity':
        distance_matrix = Metrics.diversityMatrix(mat)
        np.savetxt(savepath + filename + '_diversity.txt', distance_matrix, delimiter=',')
    else:
        distance_matrix = Metrics.NIDMatrix(mat)
        np.savetxt(savepath + filename + '_nid.txt', distance_matrix, delimiter=',')

    if manifold_type == 'MDS':
        # transform distance matrix into 2-d or 3-d coordinates to visualize
        mds2d = manifold.MDS(n_components=2, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
        mds3d = manifold.MDS(n_components=3, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
        pos2d = mds2d.fit(distance_matrix).embedding_
        pos3d = mds3d.fit(distance_matrix).embedding_
        np.savetxt(savepath + filename + '_mds2d.txt', pos2d, fmt="%.6f", delimiter=',')
        np.savetxt(savepath + filename + '_mds3d.txt', pos3d, fmt="%.6f", delimiter=',')

        # draw odm, k distribution and nmi distribution
        cv.plot_ordered_distance_matrix(distance_matrix, savepath + filename + '_original_distance.png',
                                        savepath + filename + '_odm.png')
        cv.plot_k_distribution(mat, pos2d, savepath + filename+'_k_distribution.png')
        cv.plot_nmi_max(mat, pos2d, savepath + filename + '_nmimax_distribution.png')

        # consistencies are calculated while constraints file exists.
        if constraints_file != '':
            cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_both.png',
                                consistency_type='both')
            cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_must.png',
                                consistency_type='must')
            cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_cannot.png',
                                consistency_type='cannot')
            cv.plt_consistency_corelation_with_k(mat, mlset, nlset, savepath + filename+'_normalized.png')
    return
