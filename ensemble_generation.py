import basicClusterMethods as bcm
import Cluster_Ensembles as ce
import numpy as np
import Metrics
from sklearn import cluster
from sklearn import manifold
import random as rand
import os
import cluster_visualization as cv
import generate_constraints_link as gcl

_sampling_methods = {'FSRSNN': bcm.FSRSNN_c, 'FSRSNC': bcm.FSRSNC_c}


def _get_file_name(name, s_Clusters, l_Clusters, FSR, FSR_l, SSR, SSR_l, n_members, f_stable, s_stable, method):
    """
    get file name to store the matrix (internal use only)
    """
    if not f_stable and FSR > 1.0:
        f_string = str(int(FSR_l)) + '~' + str(int(FSR))
    elif not f_stable and FSR <= 1.0:
        f_string = str(FSR_l) + '~' + str(FSR)
    elif f_stable and FSR > 1.0:
        f_string = str(int(FSR))
    else:
        f_string = str(FSR)

    if not s_stable and SSR > 1.0:
        s_string = str(int(SSR_l)) + '~' + str(int(SSR))
    elif not f_stable and FSR <= 1.0:
        s_string = str(SSR_l) + '~' + str(SSR)
    elif s_stable and SSR > 1.0:
        s_string = str(int(SSR))
    else:
        s_string = str(SSR)
    return name + '_' + str(s_Clusters) + '-' + str(l_Clusters) + '_' + s_string + '_' + f_string + \
           '_' + str(n_members) +'_' + method


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
        FSR_l = 0.05
        SSR_l = 0.1
        sampling_method = 'FSRSNN'

        if 'method' in paramSettings[name]:
            sampling_method = paramSettings[name]['method']
            if sampling_method not in _sampling_methods.keys():
                raise ValueError('ensemble generation : Method should be either \'FSRSNN\' or \'FSRSNC\'')

        if 'constraints' in paramSettings[name]:
            constraints_file = paramSettings[name]['constraints']
            mlset, nlset = gcl.read_constraints(constraints_file)
        else:
            constraints_file = ''
            mlset = []
            nlset = []

        # get parameters from dictionary if available
        if 'FSR' in paramSettings[name]:
            FSR = paramSettings[name]['FSR']
        if 'SSR' in paramSettings[name]:
            SSR = paramSettings[name]['SSR']
        if 'FSR_L' in paramSettings[name]:
            FSR_l = paramSettings[name]['FSR_L']
        if 'SSR_L' in paramSettings[name]:
            SSR_l = paramSettings[name]['SSR_L']

        if 'small_Clusters' in paramSettings[name] and 'large_Clusters' in paramSettings[name]:
            s_Clusters = int(paramSettings[name]['small_Clusters'])
            l_Clusters = int(paramSettings[name]['large_Clusters'])

        f_stable_sample = True
        s_stable_sample = True
        if 'F_STABLE' in paramSettings[name]:
            f_stable_sample = paramSettings[name]['F_STABLE']
        if 'S_STABLE' in paramSettings[name]:
            s_stable_sample = paramSettings[name]['S_STABLE']

        if FSR_l > FSR:
            FSR_l = FSR / 2
        if SSR_l > SSR:
            SSR_l = SSR / 2

        # there should be at least 2 clusters in the clustering
        if s_Clusters < 2:
            s_Clusters = 2
        if l_Clusters < s_Clusters:
            l_Clusters = s_Clusters

        tag = True

        # matrix to store clustering results
        mat = np.empty(data.shape[0])

        # generate ensemble members
        for i in range(0, n_members):
            # determine k randomly
            cluster_num = np.random.randint(s_Clusters, l_Clusters + 1)
            random_state = np.random.randint(0, 2147483647 - 1)

            cur_FSR = FSR
            cur_SSR = SSR
            if not f_stable_sample:
                cur_FSR = rand.uniform(FSR_l, FSR)
            if not s_stable_sample:
                cur_SSR = rand.uniform(SSR_l, SSR)

            # generate ensemble member by given method
            result = _sampling_methods[sampling_method](data, target, r_clusters=cluster_num,
                                                        r_state=random_state, fsr=cur_FSR, ssr=cur_SSR)
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
        filename = _get_file_name(name, s_Clusters, l_Clusters, FSR, FSR_l, SSR, SSR_l, n_members,
                                  f_stable_sample, s_stable_sample, sampling_method)
        print 'Dataset ' + name + ', consensus finished, results are saving to file : ' + filename

        # write results to external file, use %d to keep integer part only
        np.savetxt(savepath + filename + '.res', mat, fmt='%d', delimiter=',')

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
                distance_matrix = Metrics.diversityMatrix(mat)
                np.savetxt(savepath + filename + '_diversity.txt', distance_matrix, delimiter=',')
            else:
                distance_matrix = Metrics.NIDMatrix(mat)
                np.savetxt(savepath + filename + '_nid.txt', distance_matrix, delimiter=',')

            # save performances
            perf = np.array([nmi_CSPA, nmi_HGPA, nmi_MCLA, kmnmi])
            np.savetxt(savepath + filename + '_performance.txt', perf, fmt='%.6f', delimiter=',')

        if manifold_type == 'MDS':
            # transform distance matrix into 2-d or 3-d coordinates to visualize
            mds2d = manifold.MDS(n_components=2, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
            mds3d = manifold.MDS(n_components=3, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
            pos2d = mds2d.fit(distance_matrix).embedding_
            pos3d = mds3d.fit(distance_matrix).embedding_
            np.savetxt(savepath + filename + '_mds2d.txt', pos2d, fmt="%.6f", delimiter=',')
            np.savetxt(savepath + filename + '_mds3d.txt', pos3d, fmt="%.6f", delimiter=',')

            cv.draw_ordered_distance_matrix(distance_matrix, savepath + filename+'_original_distance.png',
                                            savepath + filename+'_odm.png')
            cv.plot_k_distribution(mat, pos2d, savepath + filename+'_k_distribution.png')
            if constraints_file != '':
                cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_both.png',
                                    consistency_type='both')
                cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_must.png',
                                    consistency_type='must')
                cv.plot_consistency(mat, pos2d, mlset, nlset, savepath + filename+'_consistency_cannot.png',
                                    consistency_type='cannot')
    return
