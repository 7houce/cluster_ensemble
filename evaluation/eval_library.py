import numpy as np
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as Metrics
import csv
import os

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY}


def evaluate_library(name, path, class_num, target, evaluate_methods=['CSPA']):
    labels = np.loadtxt(path + name, delimiter=',')
    if not name.endswith('_pure.res'):
        labels = labels[0:-5]
    scores = []
    for method in evaluate_methods:
        ensemble_label = _ensemble_method[method](labels, N_clusters_max=class_num)
        scores.append(Metrics.normalized_max_mutual_info_score(target, ensemble_label))
    return scores


def evaluate_libraries_to_file(names, path, class_num, target, filename, evaluate_methods=['CSPA']):
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


def do_eval_in_folder(prefix, folder, class_num, target, filename, evaluate_methods=['CSPA']):
    """
    merge all performances in specific folder into a csv file.

    :param folder:
    :param stat_file_name:
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


