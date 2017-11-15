import numpy as np
import ensemble.Cluster_Ensembles as ce
import ensemble.spectral_ensemble as spec
import evaluation.Metrics as Metrics
import os

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.spectral_ensemble}

_default_evaluate_methods = ['CSPA', 'Spectral']


def evaluate_library(name, path, class_num, target, evaluate_methods=_default_evaluate_methods):
    """
    do evaluation for a given library

    :param name: name of the library
    :param path: path where the library is
    :param class_num: #real_classes
    :param target: real class label
    :param evaluate_methods: consensus functions used for evaluation

    Return
    ------
    :return: score of all consensus functions in a list
    """
    labels = np.loadtxt(path + name, delimiter=',')
    if not name.endswith('_pure.res'):
        labels = labels[0:-5]
    scores = []
    for method in evaluate_methods:
        ensemble_label = _ensemble_method[method](labels, N_clusters_max=class_num)
        scores.append(Metrics.normalized_max_mutual_info_score(target, ensemble_label))
    return scores


def evaluate_libraries_to_file(names, path, class_num, target, filename,
                               evaluate_methods=_default_evaluate_methods,
                               direction='vertical'):
    """
    do library evaluation for given libraries

    Parameters
    ----------
    :param names: names of libraries to evaluate, in a list
    :param path: directory where the libraries are
    :param class_num: #real_classes
    :param target: real class label
    :param filename: name of the file to store evaluation result
    :param evaluate_methods: consensus functions used for evaluation, in a list, default to ['CSPA', 'Spectral']
    :param direction: horizontal or vertical output format

    """
    header = ['LibraryName']
    header.extend(evaluate_methods)
    result = np.array(header)
    for name in names:
        row = [name]
        scores = evaluate_library(name, path, class_num, target, evaluate_methods=evaluate_methods)
        row.extend(scores)
        result = np.vstack([result, np.array(row)])
    if direction == 'vertical':
        result = result.transpose()
    np.savetxt(filename, result, fmt='%s', delimiter=',')
    return


def do_eval_in_folder(prefix, folder, class_num, target, filename, evaluate_methods=_default_evaluate_methods):
    """
    do library evaluation for given libraries(with a specific prefix) in a folder
    (Deprecated)

    :param prefix:
    :param folder:
    :param class_num:
    :param target:
    :param filename:
    :param evaluate_methods:
    :return:
    """
    names = []
    if not os.path.isdir(folder):
        raise Exception("first argument should be a folder")
    for f in os.listdir(folder):
        fname = os.path.splitext(f)
        if not fname[0].endswith(prefix):
            continue
        print f
        names.append(f)
    evaluate_libraries_to_file(names, folder, class_num, target, filename, evaluate_methods=evaluate_methods)
    return


