"""
generate a library of base clustering, including k-means/constrained methods.
Author: Zhijie Lin
"""
from __future__ import print_function
from sklearn import cluster
from sklearn import manifold
import os
import time
import numpy as np
import random as rand
import subspace as sm
import ensemble.Cluster_Ensembles as ce
import utils.cluster_visualization as cv
import utils.io_func as io_func
import utils.exp_datasets as data_info
import constrained_methods.efficient_cop_kmeans as eck
import constrained_methods.constrained_clustering as cc
import evaluation.Metrics as Metrics
import evaluation.eval_library as el
import utils.settings as settings

# from global setting file.
_default_eval_path = settings.default_eval_path
_default_constraints_folder = settings.default_constraints_folder
_default_result_path = settings.default_library_path
_default_constraints_postfix = settings.default_constraints_folder

# specific default settings.
_default_FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_default_SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]

_sampling_methods = {'FSRSNN': sm.FSRSNN,
                     'FSRSNC': sm.FSRSNC}

_constrained_methods = {'E2CP': cc.E2CP,
                        'Cop_KMeans': eck.cop_kmeans_wrapper}

_INT_MAX = 2147483647


def _get_constraints_file_names(dataset_name, additional_postfix, postfixes=None):
    """
    get names of constraint file.
    format: [FOLDER_NAME] + [DATASET_NAME] + [COMMON_POSTFIX] + [ADDITIONAL_POSTFIX] + .txt
    (internal use only)
    """
    use_postfixes = postfixes if postfixes is not None else _default_constraints_postfix
    return map(lambda x: _default_constraints_folder + dataset_name + '_' + x + additional_postfix + '.txt',
               use_postfixes)


def _get_file_name(dataset_name, n_cluster_lower_bound, n_cluster_upper_bound,
                   feature_sampling, feature_sampling_lower_bound,
                   sample_sampling, sample_sampling_lower_bound,
                   n_members, f_stable, s_stable, sampling_method, is_constraint_method=False, constraint_file=None):
    """
    get file name to store the library (in a matrix)
    format: [DATASET_NAME] + [RANGE_OF_K] + [FEATURE SAMPLING] +
            [INSTANCE SAMPLING] + [#MEMBERS] + [METHOD] + *[CONSTRAINT_FILE]
    (internal use only)
    """
    # feature sampling rate
    if not f_stable and feature_sampling > 1.0:
        f_string = str(int(feature_sampling_lower_bound)) + '~' + str(int(feature_sampling))
    elif not f_stable and feature_sampling <= 1.0:
        f_string = str(feature_sampling_lower_bound) + '~' + str(feature_sampling)
    elif f_stable and feature_sampling > 1.0:
        f_string = str(int(feature_sampling))
    else:
        f_string = str(feature_sampling)

    # instance sampling rate
    if not s_stable and sample_sampling > 1.0:
        s_string = str(int(sample_sampling_lower_bound)) + '~' + str(int(sample_sampling))
    elif not f_stable and feature_sampling <= 1.0:
        s_string = str(sample_sampling_lower_bound) + '~' + str(sample_sampling)
    elif s_stable and sample_sampling > 1.0:
        s_string = str(int(sample_sampling))
    else:
        s_string = str(sample_sampling)

    # name of constraint file
    constraint_file_suffix = '' if not is_constraint_method else ('_' + constraint_file.split('Constraints/')[1])

    # final name
    filename = dataset_name + '_' + str(n_cluster_lower_bound) + '-' + str(n_cluster_upper_bound) + '_' + \
               s_string + '_' + f_string + '_' + str(n_members) + '_' + sampling_method + constraint_file_suffix
    return filename


def generate_libs_by_sampling_rate(dataset_name,
                                   n_members,
                                   cluster_lower_bound=0,
                                   cluster_upper_bound=0,
                                   fsrs=_default_FSRs,
                                   ssrs=_default_SSRs,
                                   sampling_method='FSRSNC',
                                   generate_only=True,
                                   do_eval=True,
                                   path=_default_result_path):
    """
    generating a series of libs using different sampling ratio
    used for subspace libraries generation.

    Parameters
    ----------
    :param dataset_name: name of the dataset, should be defined in exp_datasets, required
    :param n_members: #members, in integer, required
    :param cluster_lower_bound: lower bound of the #clusters, default to 5k
    :param cluster_upper_bound: upper bound of the #clusters, default to 10k
    :param fsrs: feature sampling rates (in a list), default : [0.1 .... 0.9]
    :param ssrs: instance sampling rates (in a list), default : [0.5 ... 1]
    :param sampling_method: FSRSNN or FSRSNC supported, default : FSRSNC
    :param generate_only: whether do visualization analysis or not, default to True ( only conduct generation )
    :param do_eval: do evaluation on generated dataset or not, default to True
    :param path: place to store the generated libraries, default to 'Result/'
    """
    # get the dataset and range of k
    data, target = data_info.dataset[dataset_name]['data']()
    class_num = data_info.dataset[dataset_name]['k']
    n_cluster_lower_bound = 5 * class_num if cluster_lower_bound == 0 else cluster_lower_bound
    n_cluster_upper_bound = 10 * class_num if cluster_upper_bound == 0 else cluster_upper_bound

    # generate libraries by invoking "generate_library" iteratively using all possible combination of sampling rates.
    # if the library is already exist, it will return the name of the library but not generating a new one.
    # p.s. "exist" means that there is a library file which has a same name. (see function _get_file_name)
    res_names = []
    for fsr in fsrs:
        for ssr in ssrs:
            resname = generate_library(data, target, dataset_name, n_members, class_num,
                                       n_cluster_lower_bound=n_cluster_lower_bound,
                                       n_cluster_upper_bound=n_cluster_upper_bound,
                                       feature_sampling=fsr, sample_sampling=ssr,
                                       sampling_method=sampling_method, generate_only=generate_only,
                                       path=path)
            res_names.append(resname)

    # evaluate generated libraries and store the result to a file
    # evaluation result will be store at _default_eval_path, with the name of the dataset and timestamp.
    if do_eval:
        el.evaluate_libraries_to_file(res_names,
                                      path + dataset_name + '/',
                                      class_num,
                                      target,
                                      _default_eval_path + dataset_name + time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())) + '.csv')
    return


def generate_libs_by_constraints(dataset_name,
                                 n_members,
                                 postfixes=None,
                                 additional_postfix='',
                                 cluster_lower_bound=0,
                                 cluster_upper_bound=0,
                                 member_method='E2CP',
                                 do_eval=True,
                                 path=_default_result_path):
    """
    generating a series of libs using different constraints files
    used for semi-supervised clustering.

    Parameters
    ----------
    :param dataset_name: name of the dataset, should be defined in exp_datasets, required
    :param n_members: #members, in integer, required
    :param postfixes: postfix of constraint files, default is [constraints_quarter_n, constraints_half_n ........]
    :param additional_postfix: additional postfix of constraint files, default is empty.
    :param cluster_lower_bound: lower bound of the #clusters, default to 5k
    :param cluster_upper_bound: upper bound of the #clusters, default to 10k
    :param member_method: 'Cop_KMeans' and 'E2CP' supported, default to 'E2CP'
    :param do_eval: do evaluation on generated dataset or not, default to True
    :param path: place to store the generated libraries, default to 'Result/'
    """
    # get the dataset and range of k
    data, target = data_info.dataset[dataset_name]['data']()
    class_num = data_info.dataset[dataset_name]['k']
    n_cluster_lower_bound = 5 * class_num if cluster_lower_bound == 0 else cluster_lower_bound
    n_cluster_upper_bound = 10 * class_num if cluster_upper_bound == 0 else cluster_upper_bound

    # get name of constraint files used.
    constraints_files = _get_constraints_file_names(dataset_name, additional_postfix, postfixes=postfixes)

    # generate libraries by invoking "generate_library" iteratively using all constraint files.
    # if the library is already exist, it will return the name of the library but not generating a new one.
    # p.s. "exist" means that there is a library file which has a same name. (see function _get_file_name)
    res_names = []
    for constraints_file in constraints_files:
        res_name = generate_library(data, target, dataset_name, n_members, class_num,
                                    n_cluster_lower_bound=n_cluster_lower_bound,
                                    n_cluster_upper_bound=n_cluster_upper_bound,
                                    sampling_method=member_method, constraints_file=constraints_file)
        res_names.append(res_name)

    # evaluate generated libraries and store the result to a file
    # evaluation result will be store at _default_eval_path, with the name of the dataset and timestamp.
    if do_eval:
        el.evaluate_libraries_to_file(res_names, path + dataset_name + '/', class_num, target,
                                      _default_eval_path + dataset_name + time.strftime('%Y-%m-%d_%H_%M_%S',
                                      time.localtime(time.time())) + '_' + member_method + '_' + additional_postfix + '.csv')
    return


def generate_library(data, target, dataset_name, n_members, class_num,
                     n_cluster_lower_bound=0, n_cluster_upper_bound=0,
                     feature_sampling=1.0, sample_sampling=0.7,
                     feature_sampling_lower_bound=0.05, sample_sampling_lower_bound=0.1,
                     f_stable_sample=True, s_stable_sample=True,
                     constraints_file=None, sampling_method='FSRSNC', verbose=True, path=_default_result_path,
                     metric='nid', manifold_type='MDS', subfolder=True,
                     generate_only=True):
    """
    generate a single library of ensemble member.

    Parameters
    ----------
    :param data: dataset in a ndarray
    :param target: target in a ndarray or list
    :param dataset_name: name of dataset
    :param n_members: #clusters
    :param class_num: #real_class
    :param n_cluster_lower_bound: lower bound of k
    :param n_cluster_upper_bound: upper bound of k
    :param feature_sampling: fixed sampling rate of feature, or upper bound if not stable
    :param sample_sampling:  fixed sampling rate of instances, or upper bound if not stable
    :param feature_sampling_lower_bound: lower bound of sampling rate of feature, only available if not stable
    :param sample_sampling_lower_bound: lower bound of sampling rate of instance, only available if not stable
    :param f_stable_sample: stable feature sampling or not
    :param s_stable_sample: stable instance sampling or not
    :param constraints_file: name of constraint file, only available when
    :param sampling_method: 'FSRSNC' and 'FSRSNN' supported
    :param verbose: print debug info.
    :param path: path to store the library
    :param metric: used for visualization only
    :param manifold_type: used for visualization only
    :param subfolder: save library in a separated sub-folder or not.

    Return
    ------
    :return: name of the library generated (the library itself will be stored as a file)
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
        print ('[Library Generation] : library already exists.')
        return filename+'.res'
    elif os.path.isfile(savepath + filename + '_pure.res'):
        print ('[Library Generation] : corresponding pure library already exists.')
        return filename+'_pure.res'

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

    if generate_only or is_constrained:
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
