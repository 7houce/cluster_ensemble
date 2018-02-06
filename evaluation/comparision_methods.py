import time
import csv
from sklearn import cluster
import numpy as np
import utils.exp_datasets as exd
import evaluation.Metrics as metrics
import constrained_methods.efficient_cop_kmeans as eck
import constrained_methods.constrained_clustering as cc
import utils.io_func as io_func
import utils.settings as setting
import ensemble.Cluster_Ensembles as ce

_constrained_methods = {'E2CP': cc.E2CP,
                        'Cop_KMeans': eck.cop_kmeans_wrapper}
_ensemble_methods = {'HGPA': ce.cluster_ensembles_HGPAONLY,
                     'MCLA': ce.cluster_ensembles_MCLAONLY}
_default_eval_methods = ['E2CP', 'Cop_KMeans']
_default_ensemble_eval_methods = ['HGPA', 'MCLA']
_default_eval_path = 'Comparison/'
_default_constraints_folder = 'Constraints/'
_default_library_path = setting.default_library_path
_default_constraints_postfix = ['constraints_quarter_n', 'constraints_half_n', 'constraints_n', 'constraints_onehalf_n',
                                'constraints_2n']


def _get_default_constraints_files(dataset_name, postfix, additional_postfix):
    return map(lambda x: dataset_name + '_' + x + additional_postfix, postfix)


def comparison_ensemble_methods(dataset_name, library_name, eval_method=None):
    """
    get the performance of comparison methods (ensemble)

    :param dataset_name:
    :param library_name:
    :param eval_method:
    :return:
    """
    filename = _default_eval_path + dataset_name + '_ensemble_eval_' + time.strftime('%Y-%m-%d_%H_%M_%S',
                                                                                     time.localtime(time.time())) + '.csv'
    lib = np.loadtxt(_default_library_path + dataset_name + '/' + library_name + '.res', delimiter=',')
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(library_name)
        data, targets = exd.dataset[dataset_name]['data']()
        k = exd.dataset[dataset_name]['k']
        eval_methods = _default_ensemble_eval_methods if eval_method is None else eval_method
        print '[Ensemble Comparison]: Dataset: ' + str(dataset_name)
        print '[Ensemble Comparison]: Library: ' + str(library_name)
        print '[Ensemble Comparison]: Comparison Methods: ' + str(eval_methods)
        print '[Ensemble Comparison]: Real k is ' + str(k)
        for method in eval_methods:
            ensemble_label = _ensemble_methods[method](lib, N_clusters_max=k)
            performance = metrics.normalized_max_mutual_info_score(targets, ensemble_label)
            writer.writerow([method, str(performance)])
    return


def comparison_methods(dataset_name, constraints_files=None, additional_postfix='', eval_method=None):
    """
    get the performance of comparison methods.

    Parameters
    ----------
    :param dataset_name:
    :param constraints_files:
    :param additional_postfix:
    :param eval_method:
    """
    filename = _default_eval_path + dataset_name + '_' + time.strftime('%Y-%m-%d_%H_%M_%S',
                                                                       time.localtime(time.time())) + '.csv'
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        data, targets = exd.dataset[dataset_name]['data']()
        data = data.astype(np.double)
        k = exd.dataset[dataset_name]['k']
        km = cluster.KMeans(n_clusters=k)
        km.fit(data)
        writer.writerow(['KMeans', str(metrics.normalized_max_mutual_info_score(targets, km.labels_))])
        eval_methods = _default_eval_methods if eval_method is None else eval_method
        if constraints_files is None:
            filenames = _get_default_constraints_files(dataset_name, _default_constraints_postfix, additional_postfix)
        else:
            filenames = _get_default_constraints_files(dataset_name, constraints_files, additional_postfix)
        for filename in filenames:
            ml, cl = io_func.read_constraints(_default_constraints_folder + filename + '.txt')
            for method in eval_methods:
                if method == 'Cop_KMeans':
                    result = _constrained_methods[method](data, k, ml, cl)
                    writer.writerow([filename + '_Cop_KMeans',
                                     str(metrics.normalized_max_mutual_info_score(targets, result))])
                elif method == 'E2CP':
                    e2cp = _constrained_methods[method](data=data, ml=ml, cl=cl, n_clusters=k)
                    e2cp.fit_constrained()
                    result = e2cp.labels
                    writer.writerow([filename + '_E2CP',
                                    str(metrics.normalized_max_mutual_info_score(targets, result))])
    return
