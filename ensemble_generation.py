import basicClusterMethods as bcm
import Cluster_Ensembles as ce
import numpy as np
import sys
import Metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import manifold


def getFileName(name, s_Clusters, l_Clusters, FSR, SSR, n_members):
    """
    get file name to store the matrix
    :param name:
    :param s_Clusters:
    :param l_Clusters:
    :param FSR:
    :param SSR:
    :param n_members:
    :return:
    """
    return name + '_' + str(s_Clusters) + '-' + str(l_Clusters) + '_' + str(SSR) + '_' + str(FSR) + '_' + str(n_members)


def autoGenerationWithConsensus(dataSets, paramSettings, verbose=True, path='Results/', checkDiversity=True, paint=False, n_components=3):
    """
    generate ensemble members with consensus (CSPA, HGPA, MCLA) automatically
    :param dataSets: a dictionary that keys are dataset names and values are corresponding load methods
    :param paramSettings: a nested dictionary that keys are dataset names and values are a dictionary containing params
    :param verbose: whether to output the debug information
    :param path: path to store the result matrix
    :param checkDiversity: whether to check the diversity
    :param paint: whether to paint the relationship between solutions using MDS
    :param n_components: number of dimensions to construct using MDS, 3 default
    :return:
    """
    for name, dataset in dataSets.iteritems():

        print 'start generating dataset:' + name

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

        if 'FSR' in paramSettings[name]:
            FSR = paramSettings[name]['FSR']
        if 'SSR' in paramSettings[name]:
            SSR = paramSettings[name]['SSR']

        if 'small_Clusters' in paramSettings[name] and 'large_Clusters' in paramSettings[name]:
            s_Clusters = paramSettings[name]['small_Clusters']
            l_Clusters = paramSettings[name]['large_Clusters']

        tag = True

        # matrix to store clustering results
        mat = np.empty(data.shape[0])

        # generate ensemble members
        for i in range(0, n_members):
            cluster_num = np.random.randint(s_Clusters, l_Clusters)
            random_state = np.random.randint(0, sys.maxint - 1)
            # generate ensemble member by FS-RS-NN method
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
        temp = np.reshape(target, (1, data.shape[0]))
        mat = np.vstack([mat, np.array(temp)])

        # path and filename to write the file
        fileName = getFileName(name, s_Clusters, l_Clusters, SSR, FSR, n_members)
        print 'Dataset ' + name + ', consensus finished, results are saving to file : ' + fileName

        # write results to external file, use %d to keep integer part only
        np.savetxt(path + fileName + '.res', mat, fmt='%d', delimiter=',')

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
            diverMatrix = Metrics.diversityMatrix(mat)
            np.savetxt(path + fileName + '_diversity.txt', diverMatrix, delimiter=',')

        if paint:
            diverMatrix = Metrics.diversityMatrix(mat)
            mds = manifold.MDS(n_components=n_components, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
            pos = mds.fit(diverMatrix).embedding_
            fig = plt.figure(1, figsize=(7, 5))
            # clean the figure
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            plt.cla()
            ax.scatter(pos[0:-3, 0], pos[0:-3, 1], pos[0:-3, 2], c='blue', label='Ensemble Members')
            ax.scatter(pos[-3:, 0], pos[-3:, 1], pos[-3:, 2], c='red', label='Consensus Clustering')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.legend(loc='best', shadow=True)
            ax.set_title('Solution Distribution of dataset ' + name)
            plt.savefig(path + fileName + '.svg', format='svg', dpi=120)

    return
