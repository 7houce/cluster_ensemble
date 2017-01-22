import basicClusterMethods as bcm
import Cluster_Ensembles as ce
import numpy as np
import sys
import Metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import manifold
import os

def getFileName(name, s_Clusters, l_Clusters, FSR, SSR, n_members):
    """
    get file name to store the matrix (internal use only)
    """
    return name + '_' + str(s_Clusters) + '-' + str(l_Clusters) + '_' + str(SSR) + '_' + str(FSR) + '_' + str(n_members)


def autoGenerationWithConsensus(dataSets, paramSettings, verbose=True, path='Results/', checkDiversity=True,
                                metric='diversity', manifold_type='MDS', subfolder=False):
    """
    generate ensemble members with consensus (CSPA, HGPA, MCLA) automatically

    Parameters
    ----------
    :param dataSets: a dictionary that keys are dataset names and values are corresponding load methods
    :param paramSettings: a nested dictionary that keys are dataset names and values are a dictionary containing params
    :param verbose: whether to output the debug information
    :param path: path to store the result matrix
    :param checkDiversity: whether to check the diversity
    :param metric: which, should be either 'diversity' or 'NID'.
    :param manifold_type: which method of manifold transformation used to visualize, only 'MDS' is supported now.
    :param subfolder: whether to save the results into a sub-folder named by names (they should be created manually)

    Returns
    -------
    :return:
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    for name, dataset in dataSets.iteritems():

        print 'start generating dataset:' + name

        if subfolder:
            savepath = path + name + '/'
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
        else:
            savepath = path

        # get the dataset by load method
        data, target = dataset()

        # member and classnum must be defined in paramSettings
        n_members = paramSettings[name]['members']
        class_num = paramSettings[name]['classNum']

        # default values of A B FSR SSR
        s_Clusters = class_num
        l_Clusters = class_num * 10
        FSR = 1
        SSR = 0.7

        # get parameters from dictionary if available
        if 'FSR' in paramSettings[name]:
            FSR = paramSettings[name]['FSR']
        if 'SSR' in paramSettings[name]:
            SSR = paramSettings[name]['SSR']

        if 'small_Clusters' in paramSettings[name] and 'large_Clusters' in paramSettings[name]:
            s_Clusters = int(paramSettings[name]['small_Clusters'])
            l_Clusters = int(paramSettings[name]['large_Clusters'])

        # there should be at least 2 clusters in the clustering and 1/2 * (n_sample) clusters at most
        if s_Clusters < 2:
            s_Clusters = 2
        if l_Clusters > data.shape[0] * SSR / 2:
            l_Clusters = int(data.shape[0] * SSR / 2)
        if l_Clusters < s_Clusters:
            l_Clusters = s_Clusters

        tag = True

        # matrix to store clustering results
        mat = np.empty(data.shape[0])

        # generate ensemble members
        for i in range(0, n_members):
            cluster_num = np.random.randint(s_Clusters, l_Clusters)
            random_state = np.random.randint(0, 2147483647 - 1)
            # generate ensemble member by FS-RS-NN method
            while(int(data.shape[0]*SSR) < cluster_num):
                cluster_num = cluster_num / 2
            result = bcm.FSRSNN_c(data, target, r_clusters=cluster_num, r_state=random_state, r_FSR=FSR, r_SSR=SSR)
            # print diversity
            diver = Metrics.diversityBtw2Cluster(result, target)
            if verbose:
                print 'Member' + str(i) + ' diversity between real labels = ' + str(diver)
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

        # do consensus
        labels_CSPA = ce.cluster_ensembles_CSPAONLY(mat, N_clusters_max=class_num)
        labels_HGPA = ce.cluster_ensembles_HGPAONLY(mat, N_clusters_max=class_num)
        labels_MCLA = ce.cluster_ensembles_MCLAONLY(mat, N_clusters_max=class_num)

        if verbose:
            print 'Consensus results:'
            print labels_CSPA
            print labels_HGPA
            print labels_MCLA

        # put consensus results into the matrix
        mat = np.vstack([mat, np.reshape(labels_CSPA, (1, data.shape[0]))])
        mat = np.vstack([mat, np.reshape(labels_HGPA, (1, data.shape[0]))])
        mat = np.vstack([mat, np.reshape(labels_MCLA, (1, data.shape[0]))])

        # put real labels into the matrix
        temp = np.reshape(target, (1, data.shape[0]))
        mat = np.vstack([mat, np.array(temp)])

        # path and filename to write the file
        fileName = getFileName(name, s_Clusters, l_Clusters, SSR, FSR, n_members)
        print 'Dataset ' + name + ', consensus finished, results are saving to file : ' + fileName

        # write results to external file, use %d to keep integer part only
        np.savetxt(savepath + fileName + '.res', mat, fmt='%d', delimiter=',')

        if checkDiversity:
            clf = cluster.KMeans(n_clusters=class_num)
            clf.fit(data)
            kmlabels = clf.labels_

            # print labels and diversities (between the real labels)
            nmi_CSPA = Metrics.diversityBtw2Cluster(labels_CSPA, target)
            nmi_HGPA = Metrics.diversityBtw2Cluster(labels_HGPA, target)
            nmi_MCLA = Metrics.diversityBtw2Cluster(labels_MCLA, target)
            print 'consensus result diversity (CSPA) =' + str(nmi_CSPA)
            print 'consensus diversity (HGPA) =' + str(nmi_HGPA)
            print 'consensus diversity (MCLA) =' + str(nmi_MCLA)

            kmnmi = Metrics.diversityBtw2Cluster(kmlabels, target)
            print 'single-model diversity (K-means) =' + str(kmnmi)
            if metric == 'diversity':
                diverMatrix = Metrics.diversityMatrix(mat)
                np.savetxt(savepath + fileName + '_diversity.txt', diverMatrix, delimiter=',')
            else:
                diverMatrix = Metrics.NIDMatrix(mat)
                np.savetxt(savepath + fileName + '_nid.txt', diverMatrix, delimiter=',')

            # save performances
            perf = np.array([nmi_CSPA, nmi_HGPA, nmi_MCLA, kmnmi])
            np.savetxt(savepath + fileName + '_performance.txt', perf, fmt='%.6f', delimiter=',')

        if manifold_type == 'MDS':
            # transform distance matrix into 2-d or 3-d coordinates to visualize
            mds2d = manifold.MDS(n_components=2, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
            mds3d = manifold.MDS(n_components=3, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
            pos2d = mds2d.fit(diverMatrix).embedding_
            pos3d = mds3d.fit(diverMatrix).embedding_
            np.savetxt(savepath + fileName + '_mds2d.txt', pos2d, fmt="%.6f", delimiter=',')
            np.savetxt(savepath + fileName + '_mds3d.txt', pos3d, fmt="%.6f", delimiter=',')

            # save 2-d figure only
            fig = plt.figure(1)
            plt.clf()
            ax = plt.axes([0., 0., 1., 1.])
            plt.cla()
            plt.scatter(pos2d[0:-4, 0], pos2d[0:-4, 1], c='blue', label='Ensemble Members')
            plt.scatter(pos2d[-4:-1, 0], pos2d[-4:-1, 1], c='red', label='Consensus Clustering')
            plt.scatter(pos2d[-1:, 0], pos2d[-1:, 1], c='yellow', label='Real')
            plt.legend(loc='best', shadow=True)
            plt.title(name)
            plt.savefig(savepath + fileName + '_mds2d_vis.svg', format='svg', dpi=120)

    return
