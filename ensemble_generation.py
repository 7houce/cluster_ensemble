import basicClusterMethods as bcm
import Cluster_Ensembles as ce
import numpy as np
import sys
import Metrics
from sklearn import cluster


def getFileName(name, A, B, SSR, FSR, n_members):
    """
    get file name to store the matrix
    :param name:
    :param A:
    :param B:
    :param SSR:
    :param FSR:
    :param n_members:
    :return:
    """
    return name + '_' + str(A) + '-' + str(B) + '_' + str(SSR) + '_' + str(FSR) + '_' + str(n_members) + '.res'


def autoGenerationWithConsensus(dataSets, paramSettings, verbose=True, path='Results/', checkdiversity=True):
    """
    generate ensemble members with consensus (CSPA, HGPA, MCLA) automatically
    :param dataSets: a dictionary that keys are dataset names and values are corresponding load methods
    :param paramSettings: a nested dictionary that keys are dataset names and values are a dictionary containing params
    :param verbose: whether to output the debug information
    :param path: path to store the result matrix
    :param checkdiversity: whether to check the diversity
    :return:
    """
    for name, dataset in dataSets.iteritems():

        print 'start generating dataset:'+name

        # get the dataset by load method
        data, target = dataset()

        # member and classnum must be defined in paramSettings
        n_members = paramSettings[name]['members']
        classnum = paramSettings[name]['classnum']

        # default values of A B FSR SSR
        A = classnum
        B = classnum * 10
        FSR = 1
        SSR = 0.7

        if 'FSR' in paramSettings[name]:
            FSR = paramSettings[name]['FSR']
        if 'SSR' in paramSettings[name]:
            SSR = paramSettings[name]['SSR']

        if 'A' in paramSettings[name] and 'B' in paramSettings[name]:
            A = paramSettings[name]['A']
            B = paramSettings[name]['B']

        tag = True

        # matrix to store clustering results
        mat = np.empty(data.shape[0])

        # generate ensemble members
        for i in range(0, n_members):
            cluster_num = np.random.randint(A, B)
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
        labels_CSPA = ce.cluster_ensembles_CSPAONLY(mat, N_clusters_max=classnum)
        labels_HGPA = ce.cluster_ensembles_HGPAONLY(mat, N_clusters_max=classnum)
        labels_MCLA = ce.cluster_ensembles_MCLAONLY(mat, N_clusters_max=classnum)

        if verbose:
            print 'Consensus results:'
            print labels_CSPA
            print labels_HGPA
            print labels_MCLA

        # put consensus results into the matrix
        mat = np.vstack([mat, np.reshape(labels_CSPA, (1, data.shape[0]))])
        mat = np.vstack([mat, np.reshape(labels_HGPA, (1, data.shape[0]))])
        mat = np.vstack([mat, np.reshape(labels_MCLA, (1, data.shape[0]))])

        # path and filename to write the file
        fileName = getFileName(name, A, B, SSR, FSR, n_members)
        print 'Dataset ' + name + ', consensus finished, results are saving to file : ' + fileName

        # write results to external file, use %d to keep integer part only
        np.savetxt(path + fileName, mat, fmt='%d', delimiter=',')

        if checkdiversity:
            clf = cluster.KMeans(n_clusters=classnum)
            clf.fit(data)
            kmlabels = clf.labels_

            # print labels
            nmi_CSPA = Metrics.diversityBtw2Cluster(labels_CSPA, target)
            nmi_HGPA = Metrics.diversityBtw2Cluster(labels_HGPA, target)
            nmi_MCLA = Metrics.diversityBtw2Cluster(labels_MCLA, target)
            print 'consensus result diversity (CSPA) =' + str(nmi_CSPA)
            print 'consensus diversity (HGPA) =' + str(nmi_HGPA)
            print 'consensus diversity (MCLA) =' + str(nmi_MCLA)

            kmnmi = Metrics.diversityBtw2Cluster(kmlabels, target)
            print 'single-model diversity (K-means) =' + str(kmnmi)
    return