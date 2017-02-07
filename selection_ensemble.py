from __future__ import print_function
from MSTClustering import MSTClustering
import numpy as np
import os
import matplotlib.pyplot as plt
import Cluster_Ensembles as ce
import Metrics
import generate_constraints_link as gcl

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
            mstmodel.fit(distanceMatrix[0:-4, 0:-4])

            # compute average consistency of each cluster of solutions
            avg_cons = Metrics.average_consistency(mstmodel.labels_, labels[0:-4], mlset, nlset)

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
            cluster_labels = labels[0:-4][mstmodel.labels_ == maxclu]
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
                xs = pos[0:-4][mstmodel.labels_ == i, 0]
                ys = pos[0:-4][mstmodel.labels_ == i, 1]
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


def all_cluster_consensus(nidpath, respath, distype='nid', cutoff=0.9):
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



# mlset, nlset = gcl.read_constraints('constraints.txt')
# consistency_print('Results/2.res', 'Results/2.txt', mlset, nlset, '1111.png')

# k_distribution('Results/1.res', 'Results/1.txt', 'ISOLET_k.png')
# k_distribution('Results/2.res', 'Results/2.txt', 'digit_k.png')
