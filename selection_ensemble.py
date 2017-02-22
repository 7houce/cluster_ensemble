from __future__ import print_function
from MSTClustering import MSTClustering
import numpy as np
import os
import matplotlib.pyplot as plt
import Cluster_Ensembles as ce
import Metrics
import cluster_visualization as cv
from sklearn import preprocessing

_colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod',
           'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']


def consistency_selection(nidpath, pospath, respath, savepath, mlset, nlset, distype='nid', cutoff=0.9):
    """

    :param nidpath:
    :param pospath:
    :param respath:
    :param savepath:
    :param mlset:
    :param nlset:
    :param distype:
    :param cutoff:
    :return:
    """
    nidpath = os.path.expanduser(nidpath)
    for f in os.listdir(nidpath):
        if f.startswith('.'):
            continue
        fullpath = os.path.join(nidpath, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            filename = fname[0].split('_' + distype)[0]
            dataset_name = filename.split('_')[0]
            if not os.path.isdir(savepath + dataset_name):
                os.mkdir(savepath + dataset_name)

            # read distance matrix, position matrix and label matrix from external file
            # note that the last 4 rows / cols are naive consensus & real labels
            distanceMatrix = np.loadtxt(fullpath, delimiter=',')
            pos = np.loadtxt(pospath + filename + '_mds2d.txt', delimiter=',')
            labels = np.loadtxt(respath + filename + '.res', delimiter=',')
            labels = labels.astype(int)

            # real labels store in the last row
            target = labels[-1]
            class_num = len(np.unique(target))

            # do mst clustering, we assume that there should be more than 5 solutions in each cluster
            mstmodel = MSTClustering(cutoff=cutoff, min_cluster_size=5, metric='precomputed')
            mstmodel.fit(distanceMatrix[0:-5, 0:-5])

            # compute average consistency of each cluster of solutions
            avg_cons = Metrics.average_consistency(mstmodel.labels_, labels[0:-5], mlset, nlset)

            # find the cluster of solution with largest consistency
            maxclu = 0
            max_cons = 0.0
            print (avg_cons)
            for clu, cons in avg_cons.iteritems():
                if clu == -1:
                    continue
                if cons > max_cons:
                    maxclu = clu
                    max_cons = cons

            # do consensus, note that last 4 rows should be skipped
            cluster_labels = labels[0:-5][mstmodel.labels_ == maxclu]
            labels_CSPA = ce.cluster_ensembles_CSPAONLY(cluster_labels, N_clusters_max=class_num)
            labels_HGPA = ce.cluster_ensembles_HGPAONLY(cluster_labels, N_clusters_max=class_num)
            labels_MCLA = ce.cluster_ensembles_MCLAONLY(cluster_labels, N_clusters_max=class_num)

            # print labels and diversities (between the real labels)
            nmi_CSPA = 1 - Metrics.diversityBtw2Cluster(labels_CSPA, target)
            nmi_HGPA = 1 - Metrics.diversityBtw2Cluster(labels_HGPA, target)
            nmi_MCLA = 1 - Metrics.diversityBtw2Cluster(labels_MCLA, target)
            print ('consensus result diversity (CSPA) =' + str(nmi_CSPA))
            print ('consensus diversity (HGPA) =' + str(nmi_HGPA))
            print ('consensus diversity (MCLA) =' + str(nmi_MCLA))

            # store visualization file using 2d-MDS
            fig = plt.figure(1)
            plt.clf()
            clusters = np.unique(mstmodel.labels_)
            for i in clusters:
                xs = pos[0:-5][mstmodel.labels_ == i, 0]
                ys = pos[0:-5][mstmodel.labels_ == i, 1]
                ax = plt.axes([0., 0., 1., 1.])
                if i == -1:
                    plt.scatter(xs, ys, c=_colors[((int(i) + 1) % len(_colors))], label='Outliers')
                elif i == maxclu:
                    plt.scatter(xs, ys, c=_colors[((int(i) + 1) % len(_colors))], marker='*', label='Selected')
                else:
                    plt.scatter(xs, ys, c=_colors[((int(i) + 1) % len(_colors))], label='Clusters-' + str(i))

            plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='blue', marker='D', label='Consensus')
            plt.scatter(pos[-1:, 0], pos[-1:, 1], c='red', marker='D', label='Real')
            plt.legend(loc='best', shadow=True)
            plt.savefig(savepath + dataset_name + '/' + filename + '_afterMST_selection_' + str(cutoff) + '.png',
                        format='png', dpi=240)

    return


def all_cluster_consensus(distance_matrix, labels, pos, savepath, logger, mlset, nlset, cutoff=0.9):
    """

    :param distance_matrix:
    :param labels:
    :param pos:
    :param savepath:
    :param cutoff:
    :return:
    """
    target = labels[-1]
    class_num = len(np.unique(target))

    # do mst clustering, we assume that there should be more than 2 solutions in each cluster
    mstmodel = MSTClustering(cutoff=cutoff, min_cluster_size=2, metric='precomputed')
    mstmodel.fit(distance_matrix[0:-5, 0:-5])
    filename = savepath + str(cutoff) + '.png'
    cv.plot_mst_result(mstmodel, pos, filename)

    clusters = np.unique(mstmodel.labels_)

    # compute average consistency of each cluster of solutions
    avg_cons_both = Metrics.average_consistency(mstmodel.labels_, labels[0:-5], mlset, nlset)
    avg_cons_must = Metrics.average_consistency(mstmodel.labels_, labels[0:-5], mlset, nlset, cons_type='must')
    avg_cons_cannot = Metrics.average_consistency(mstmodel.labels_, labels[0:-5], mlset, nlset, cons_type='cannot')

    for i in clusters:
        # do consensus, note that last 4 rows should be skipped
        cluster_labels = labels[0:-4][mstmodel.labels_ == i]
        labels_CSPA = ce.cluster_ensembles_CSPAONLY(cluster_labels, N_clusters_max=class_num)
        labels_HGPA = ce.cluster_ensembles_HGPAONLY(cluster_labels, N_clusters_max=class_num)
        labels_MCLA = ce.cluster_ensembles_MCLAONLY(cluster_labels, N_clusters_max=class_num)

        # print labels and diversities (between the real labels)
        nmi_CSPA = Metrics.normalized_max_mutual_info_score(labels_CSPA, target)
        nmi_HGPA = Metrics.normalized_max_mutual_info_score(labels_HGPA, target)
        nmi_MCLA = Metrics.normalized_max_mutual_info_score(labels_MCLA, target)
        logger.debug('Cluster ' + str(i) + ':')
        logger.debug('Both Consistency is ' + str(avg_cons_both[i]))
        logger.debug('Must Consistency is ' + str(avg_cons_must[i]))
        logger.debug('Cannot Consistency is ' + str(avg_cons_cannot[i]))
        logger.debug('CSPA performance is ' + str(nmi_CSPA))
        logger.debug('HGPA performance is ' + str(nmi_HGPA))
        logger.debug('MCLA performance is ' + str(nmi_MCLA))
        logger.debug('-----------------------------mst finish--------------------------------')
        logger.debug('')
    return


def all_cluster_consensus_in_file(nidpath, respath, distype='nid', cutoff=0.9):
    """

    :param nidpath:
    :param respath:
    :param distype:
    :param cutoff:
    :return:
    """
    nidpath = os.path.expanduser(nidpath)
    for f in os.listdir(nidpath):
        if f.startswith('.'):
            continue
        fullpath = os.path.join(nidpath, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            filename = fname[0].split('_' + distype)[0]
            dataset_name = filename.split('_')[0]

            # read distance matrix, position matrix and label matrix from external file
            # note that the last 4 rows / cols are naive consensus & real labels
            print (fullpath)
            distanceMatrix = np.loadtxt(fullpath, delimiter=',')
            labels = np.loadtxt(respath + filename + '.res', delimiter=',')
            labels = labels.astype(int)

            # real labels store in the last row
            target = labels[-1]
            class_num = len(np.unique(target))

            # do mst clustering, we assume that there should be more than 5 solutions in each cluster
            mstmodel = MSTClustering(cutoff=cutoff, min_cluster_size=5, metric='precomputed')
            mstmodel.fit(distanceMatrix[0:-4, 0:-4])

            clusters = np.unique(mstmodel.labels_)
            for i in clusters:
                # do consensus, note that last 4 rows should be skipped
                cluster_labels = labels[0:-4][mstmodel.labels_ == i]
                labels_CSPA = ce.cluster_ensembles_CSPAONLY(cluster_labels, N_clusters_max=class_num)
                labels_HGPA = ce.cluster_ensembles_HGPAONLY(cluster_labels, N_clusters_max=class_num)
                labels_MCLA = ce.cluster_ensembles_MCLAONLY(cluster_labels, N_clusters_max=class_num)

                # print labels and diversities (between the real labels)
                nmi_CSPA = 1 - Metrics.diversityBtw2Cluster(labels_CSPA, target)
                nmi_HGPA = 1 - Metrics.diversityBtw2Cluster(labels_HGPA, target)
                nmi_MCLA = 1 - Metrics.diversityBtw2Cluster(labels_MCLA, target)
                print ('Cluster ' + str(i) + '===========================================')
                print ('consensus result diversity (CSPA) =' + str(nmi_CSPA))
                print ('consensus diversity (HGPA) =' + str(nmi_HGPA))
                print ('consensus diversity (MCLA) =' + str(nmi_MCLA))

    return


def mst_with_cutoff(distance_matrix, pos, labels, savepath, logger, mlset, nlset, cutoff_type='rate',
                    cutoff_upper=0.9, cutoff_lower=0.1, interval=0.05, min_cluster_size=2, top_k=5):
    """
    Do several MST clustering on given library of solutions in a range of different 'cutoff' parameters
    Used for deciding the suitable cutoff value which will be utilized in the MST selection procedure

    Parameters
    ----------
    :param distance_matrix: distance matrix of given library of solutions
    :param pos: positions generated by MDS or other embedding approaches, used for visualization
    :param savepath: path where the visualization of clustering result stored
    :param cutoff_type: type of parameter 'cutoff', should either be :
                       'threshold' :  edges with larger weight than threshold will be cut
                       'rate' :       number(>=1) / fraction(<1.0) of edges that will be cut
    :param cutoff_upper: upper bound of parameter cutoff
    :param cutoff_lower: lower bound of parameter cutoff
    :param interval: interval of parameter cutoff
    :param min_cluster_size:

    """
    # do MST clustering by setting 'cutoff' between [cutoff_lower, cutoff_upper] with given interval
    cluster_num = {}
    cur_cutoff = cutoff_lower
    while cur_cutoff <= cutoff_upper:
        if cutoff_type == 'threshold':
            mstmodel = MSTClustering(cutoff_scale=cur_cutoff, min_cluster_size=min_cluster_size,
                                     metric='precomputed', approximate=False)
        else:
            mstmodel = MSTClustering(cutoff=cur_cutoff, min_cluster_size=min_cluster_size,
                                     metric='precomputed', approximate=False)
        mstmodel.fit(distance_matrix[0:-5, 0:-5])
        # filename = savepath + str(cur_cutoff) + '.png'
        cluster_num[cur_cutoff] = len(np.unique(mstmodel.labels_))
        # cv.plot_mst_result(mstmodel, pos, filename)
        cur_cutoff += interval
    sorted_cluster_num = sorted(cluster_num.iteritems(), key=lambda item: item[1], reverse=True)
    while top_k > 0:
        consensus_cutoff = sorted_cluster_num[top_k-1][0]
        all_cluster_consensus(distance_matrix, labels, pos, savepath, logger, mlset, nlset, cutoff=consensus_cutoff)
        top_k -= 1
    return


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
