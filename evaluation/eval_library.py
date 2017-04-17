import numpy as np
import ensemble.Cluster_Ensembles as ce
import ensemble.spectral_ensemble as spec
import evaluation.Metrics as Metrics
import csv
import os

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.spectral_ensemble}

_default_evaluate_methods = ['CSPA', 'Spectral']


def evaluate_library(name, path, class_num, target, evaluate_methods=_default_evaluate_methods):
    """
    do evaluation for a given library

    :param name:
    :param path:
    :param class_num:
    :param target:
    :param evaluate_methods:
    :return:
    """
    labels = np.loadtxt(path + name, delimiter=',')
    if not name.endswith('_pure.res'):
        labels = labels[0:-5]
    scores = []
    for method in evaluate_methods:
        ensemble_label = _ensemble_method[method](labels, N_clusters_max=class_num)
        scores.append(Metrics.normalized_max_mutual_info_score(target, ensemble_label))
    return scores


def evaluate_libraries_to_file(names, path, class_num, target, filename, evaluate_methods=_default_evaluate_methods):
    """
    do library evaluation for given libraries

    :param names:
    :param path:
    :param class_num:
    :param target:
    :param filename:
    :param evaluate_methods:
    :return:
    """
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        header = ['LibraryName']
        header.extend(evaluate_methods)
        writer.writerow(header)
        for name in names:
            row = [name]
            scores = evaluate_library(name, path, class_num, target, evaluate_methods=evaluate_methods)
            row.extend(scores)
            writer.writerow(row)
    return


def do_eval_in_folder(prefix, folder, class_num, target, filename, evaluate_methods=_default_evaluate_methods):
    """
    do library evaluation for given libraries(with a specific prefix) in a folder

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


