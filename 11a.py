import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import datetime
import pickle
from sklearn import cluster
from sklearn import metrics
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import random as rand
import os
from sklearn import preprocessing
import logger_module as lm
import scipy
import sklearn.cluster.k_means_
import gc
import utils.load_dataset as ld
import ensemble.spectral_ensemble as spec
from sklearn import metrics

# colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod', 'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']


"""
Load result data and generate the diversity matrix
"""
# mat = np.loadtxt('Results/digit_10-100_0.5_0.7_160.res', delimiter=',')
# data, target = dataSetPreprocessing.loadDigits()
# print target
# temp = np.reshape(target, (1, data.shape[0]))
# mat = np.vstack([mat, np.array(temp)])
# print mat
# diverMatrix = Metrics.diversityMatrix(mat)
# np.savetxt('diversity.txt', diverMatrix, delimiter=',', fmt='%.6f')


"""
Show histogram of given diversity matrix
"""
# diverMatrix = np.loadtxt('Results/Wine/Wine_3-30_0.7_0.7_160_nid.txt', delimiter=',')
# diverMatrix = np.reshape(diverMatrix, (1, diverMatrix.shape[0]**2))
# plt.hist(diverMatrix.T, bins='auto')
# plt.show()


"""
Conduct MST Clustering on given diversity matrix
"""
# diverMatrix = np.loadtxt('Results/Wine/Wine_3-30_0.7_0.7_160_nid.txt', delimiter=',')
# pos = np.loadtxt('Results/Wine/Wine_3-30_0.7_0.7_160_mds2d.txt', delimiter=',')
# model = MSTClustering(cutoff_scale=0.4, min_cluster_size=5, metric='precomputed')
# model.fit(diverMatrix[0:-4, 0:-4])
# np.savetxt('aaa.txt', model.labels_, fmt="%d", delimiter=',')
# clusters = np.unique(model.labels_)
# for i in clusters:
#     xs = pos[0:-4][model.labels_ == i, 0]
#     ys = pos[0:-4][model.labels_ == i, 1]
#     fig = plt.figure(1)
#     ax = plt.axes([0., 0., 1., 1.])
#     if i != -1:
#         plt.scatter(xs, ys, c=colors[int(i)+1], label='Clusters-'+str(i))
#     else:
#         plt.scatter(xs, ys, c=colors[int(i) + 1], label='Outliers')
# plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='maroon', label='Consensus Clustering')
# plt.scatter(pos[-1:, 0], pos[-1:, 1], c='lawngreen', label='Real')
# plt.legend(loc='best', shadow=True)
# plt.show()


# print diverMatrix



"""
Do MDS transformation and plot 3D graph.
"""
# mds = manifold.MDS(n_components=3, max_iter=10000, eps=1e-12, dissimilarity='precomputed')
# pos = mds.fit(diverMatrix).embedding_
# fig = plt.figure(1, figsize=(7, 5))
# # clean the figure
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# plt.cla()
# ax.scatter(pos[0:-3, 0], pos[0:-3, 1], pos[0:-3, 2], c='blue', label='Ensemble Members')
# ax.scatter(pos[-4:-3, 0], pos[-4:-3, 1], pos[-4:-3, 2], c='black', label='CSPA')
# ax.scatter(pos[-3:-2, 0], pos[-3:-2, 1], pos[-3:-2, 2], c='red', label='HGPA')
# ax.scatter(pos[-2:-1, 0], pos[-2:-1, 1], pos[-2:-1, 2], c='green', label='MCLA')
# ax.scatter(pos[-1:, 0], pos[-1:, 1], pos[-1:, 2], c='yellow', label='Real')
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# ax.legend(loc='best', shadow=True)
# ax.set_title('Solution Distribution of dataset')
# plt.savefig('aaaa' + '.svg', format='svg', dpi=120)
# plt.show()
#
# np.savetxt('pos.txt', pos, delimiter=',', fmt='%.6f')


"""
Test NID matrix calculation
"""
# mat = np.loadtxt('Results/digit_20-200_1_0.7_20.res', delimiter=',')
# # temp = np.reshape(target, (1, data.shape[0]))
# # mat = np.vstack([mat, np.array(temp)])
# print mat
# t1 = time.clock()
# nidMatrix = Metrics.NIDMatrix(mat)
# t2 = time.clock()
# print 'NID done in ' + str(t2 - t1) + ' seconds'
# t1 = time.clock()
# divMatrix = Metrics.diversityMatrix(mat)
# t2 = time.clock()
# print 'DIV done in ' + str(t2 - t1) + ' seconds'
# np.savetxt('nid.txt', nidMatrix, delimiter=',', fmt='%.6f')
# np.savetxt('div.txt', divMatrix, delimiter=',', fmt='%.6f')


"""
test single clustering
"""
# dataSet, target = dataSetPreprocessing.loadIonosphere()
# print dataSet.shape
# print dataSet
# print target
# print target.shape
# clf = cluster.KMeans(n_clusters=2,
#                      verbose=0)
# clf.fit(dataSet)
# kmlabels = clf.labels_
# print kmlabels.shape
# print kmlabels
# # kmnmi = Metrics.diversityBtw2Cluster(kmlabels, target)
# ari = metrics.homogeneity_score(target, kmlabels)
# print 'ARI (K-means) =' + str(ari)

"""
Load clustering library (names)
"""
library_folder = 'Results/ISOLET/'
library_name0 = 'ISOLET_26-260_0.5_0.3_160_FSRSNC'
library_name1 = 'ISOLET_16-36_0.5_0.3_160_FSRSNC'
library_name2 = 'ISOLET_26-26_0.5_0.3_160_FSRSNC'

"""
plot library
"""
# mlset, nlset = gcl.read_constraints('Constraints/N_constraints.txt')
# cv.plot_for_all_library(library_folder, type='k')
# cv.plot_for_all_library(library_folder, type='nmi')
# cv.plot_for_all_library(library_folder, type='odm')
# cv.plot_for_all_library(library_folder, type='cons_both', mlset=mlset, nlset=nlset)
# cv.plot_for_all_library(library_folder, type='cons_must', mlset=mlset, nlset=nlset)
# cv.plot_for_all_library(library_folder, type='cons_cannot', mlset=mlset, nlset=nlset)
# cv.plot_for_all_library(library_folder, type='n_corelation', mlset=mlset, nlset=nlset)


"""
k-value selection
"""
# stop = False
# k = 50
# while not stop:
#     logger.debug('==============K Selection for K='+str(k)+' ===============================')
#     se.k_selection_ensemble_for_library(library_folder, library_name0, k, logger, weighted=True,
#                                         mlset=mlset, nlset=nlset, alpha=0.5, ctype='both')
#     k += 10
#     if k > 200:
#         stop = True

# stop = False
# threshold_tuple1 = [(0.2, 0.5), (0.2, 0.55), (0.2, 0.6), (0.2, 0.65), (0.2, 0.7),
#                    (0.22, 0.55), (0.24, 0.6), (0.26, 0.65), (0.28, 0.7)]
# threshold_tuple2 = [(0.2, 0.4), (0.2, 0.45), (0.2, 0.5), (0.2, 0.55), (0.2, 0.6),
#                     (0.3, 0.5), (0.4, 0.55), (0.4, 0.6)]
# logger.debug('Results for:'+library_name1)
# for tuple in threshold_tuple1:
#     logger.debug('==============Consistency Selection for Consistency=Must>' + str(tuple[0]) + ',Cannot>'+str(tuple[1])+' ===============================')
#     se.consistency_selection_ensemble_for_library(library_folder, library_name1, mlset, nlset, logger, tuple[0], tuple[1])

# logger.debug('Results for:'+library_name2)
# for tuple in threshold_tuple2:
#     logger.debug('==============Consistency Selection for Consistency=Must>' + str(tuple[0]) + ',Cannot>'+str(tuple[1])+' ===============================')
#     se.consistency_selection_ensemble_for_library(library_folder, library_name2, mlset, nlset, logger, tuple[0], tuple[1])

# samples = np.array([0, 0, 0, 1, 1, 2, 2])
# samples = np.vstack([samples, np.array([1, 1, 1, 2, 2, 0, 0])])
# samples = np.vstack([samples, np.array([0, 0, 1, 1, 2, 2, 2])])
# samples = np.vstack([samples, np.array([0, 1, np.NAN, 0, 1, np.NAN, np.NAN])])
#
# labels = ce.cluster_ensembles_MCLAONLY(samples)
#
# print labels


"""
Consistency Selection
"""
# names = ['ISOLET_26-26_0.5_0.3_160_FSRSNC', 'ISOLET_16-36_0.5_0.3_160_FSRSNC']
# cons_type = ['must', 'cannot']
# for ctype in cons_type:
#     for name in names:
#         logger.debug('-----------------'+str(name)+' '+ctype+'-----------------------')
#         labels = np.loadtxt('Results/ISOLET/'+name+'.res', delimiter=',')
#         labels = labels.astype(int)
#         con_per_cluster = []
#         con_clustering = []
#         t1 = time.clock()
#         for label in labels[0:-5]:
#             con_per_cluster.append(Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=ctype))
#         for label in labels[0:-5]:
#             con_clustering.append(Metrics.consistency(label, mlset, nlset, cons_type=ctype))
#         t2 = time.clock()
#         ass = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#         for alpha in ass:
#             cspa_labels = ce.cluster_ensembles_MCLAONLY(labels[0:-5], N_clusters_max=26,
#                                                         weighted=True, clustering_weights=con_clustering,
#                                                         cluster_level_weights=con_per_cluster, alpha=alpha)
#             nmi = Metrics.normalized_max_mutual_info_score(cspa_labels, labels[-1])
#             logger.debug('MCLA ALPHA='+str(alpha)+', NMI='+str(nmi))
#         full_label = ce.cluster_ensembles_MCLAONLY(labels[0:-5], N_clusters_max=26)
#         full_nmi = Metrics.normalized_max_mutual_info_score(full_label, labels[-1])
#         logger.debug('FULL NMI='+str(full_nmi))
#         logger.debug('-------------------------------------')

"""
test spambase
"""
# data, target = dataSetPreprocessing.load_spam_base(normalized=True)
# ml, nl = gcl.read_constraints('Constraints/Spambase_constraints_500.txt')
# print ml
# print nl
# labels = mckd.cop_kmeans_wrapper(data, 2, ml, nl)
# print Metrics.normalized_max_mutual_info_score(target, labels)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(data)
# print X_train_minmax
#
# data_normed = preprocessing.normalize(data, norm='l2')
# # print data_normed
# # print data
# km = cluster.KMeans(n_clusters=2)
# km.fit(data)
# print Metrics.normalized_max_mutual_info_score(target, km.labels_)

"""
NMF
"""
# from sklearn.decomposition import NMF
# # data, target = dataSetPreprocessing.loadIsolet()
# model = NMF(n_components=2, init='random', random_state=0)
# probs = model.fit_transform(data_normed)
# # labels = model.components_
# labels = []
# for prob in probs:
#     labels.append(np.argmax(prob))
# print probs.shape
# print probs
# print Metrics.normalized_max_mutual_info_score(target, labels)
# print precision(target, labels)

# cop_labels = mckd.cop_kmeans_wrapper(data, 26, mlset, nlset)
# print cop_labels
# print Metrics.normalized_max_mutual_info_score(target, cop_labels)
# print Metrics.consistency(cop_labels, mlset, nlset)

"""
k-means ensemble
"""
# labels = []
# for i in range(0, 160):
#     km = cluster.KMeans(n_clusters=2)
#     km.fit(data)
#     labels.append(km.labels_)
# labels = np.array(labels)
# mcla_labels = ce.cluster_ensembles_HGPAONLY(labels, N_clusters_max=2)
# print np.unique(mcla_labels)
# print mcla_labels
# print len(mcla_labels[mcla_labels == 0])
# print len(mcla_labels[mcla_labels == 1])
# print Metrics.normalized_max_mutual_info_score(target, mcla_labels)
# print Metrics.precision(target, mcla_labels)


"""
do stat
"""
# Metrics.do_stat_in_folder('Results/Spambase/', 'stat.csv')


"""
mnist 4000
"""
# data, target = dataSetPreprocessing.load_mnist_4000()
# print data.shape
# data = data.astype(float)
# # km = cluster.KMeans(n_clusters=10, n_init=1)
# t1 = time.clock()
# # km.fit(data)
# labels = mckd.cop_kmeans_wrapper(data, 10, [], [])
# t2 = time.clock()
# print Metrics.normalized_max_mutual_info_score(target, labels)
# print t2-t1

"""
ISOLET
"""
# data, target = dataSetPreprocessing.loadIsolet()
# labels = []
# for i in range(0, 160):
#     km = cluster.KMeans(n_clusters=26, n_init=1)
#     # t1 = time.clock()
#     km.fit(data)
#     # t2 = time.clock()
#     labels.append(km.labels_)
#     print '1 time'
# labels = np.array(labels, dtype=int)
# cspa_labels = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=26)
# print Metrics.normalized_max_mutual_info_score(target, cspa_labels)
# print Metrics.normalized_max_mutual_info_score(target, km.labels_)
# print t2-t1

"""
subplot
"""
# fig, ax = plt.subplots(nrows=1, ncols=2)
# nid = np.loadtxt('Results/Iris/Iris_3-22_0.3_0.3_160_nid.txt', delimiter=',')
# mds2d = np.loadtxt('Results/Iris/Iris_3-22_0.3_0.3_160_mds2d.txt', delimiter=',')
# i, j = cv._spin_sts(nid)
# new_matrix = cv._permute_matrix(nid, j)
# ax[0].imshow(new_matrix, vmin=np.min(new_matrix), vmax=np.max(new_matrix))
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# ax[0].set_xlabel('$(a)$', fontsize=28)
#
# ax[1].scatter(mds2d[0:-4, 0], mds2d[0:-4, 1], c='blue')
# ax[1].scatter(mds2d[-4:-1, 0], mds2d[-4:-1, 1], c='blue')
# ax[1].scatter(mds2d[-1, 0], mds2d[-1, 1], c='blue')
# ax[1].set_xticks([])
# ax[1].set_yticks([])
# ax[1].set_xlabel('$(b)$', fontsize=28)
#
#
# plt.show()

"""
cop-kmeans, sparse data
"""
# data, target = dataSetPreprocessing.load_wap(sparse_type='csr')
# labels = mckd.cop_kmeans_wrapper(data, 20, [], [])
# print Metrics.normalized_max_mutual_info_score(target, labels)
# km = cluster.KMeans(n_clusters=20, n_init=1, verbose=5)
# km.fit(data)
# print Metrics.normalized_max_mutual_info_score(target, km.labels_)

"""
generate constraints
"""
# data, target = dataSetPreprocessing.load_mnist_4000()
# print len(data)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.25*len(data), 0.25*len(data))
# gcl.store_constraints('Constraints/MNIST4000_constraints_half_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.5*len(data), 0.5*len(data))
# gcl.store_constraints('Constraints/MNIST4000_constraints_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.75*len(data), 0.75*len(data))
# gcl.store_constraints('Constraints/MNIST4000_constraints_onehalf_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, len(data), len(data))
# gcl.store_constraints('Constraints/MNIST4000_constraints_2n.txt', ml, cl)

#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 2000, 2000)
# gcl.store_constraints('Constraints/MNIST4000_constraints_4000_3.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 2000, 2000)
# gcl.store_constraints('Constraints/MNIST4000_constraints_4000_4.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 2000, 2000)
# gcl.store_constraints('Constraints/MNIST4000_constraints_4000_5.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 2000, 2000)
# gcl.store_constraints('Constraints/MNIST4000_constraints_4000_6.txt', ml, cl)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_5.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_6.txt')
# print Metrics.consistency(target, ml, cl)

# data, target = dataSetPreprocessing.load_coil20()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.25*len(data), 0.25*len(data))
# gcl.store_constraints('Constraints/COIL20_constraints_half_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.5*len(data), 0.5*len(data))
# gcl.store_constraints('Constraints/COIL20_constraints_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.75*len(data), 0.75*len(data))
# gcl.store_constraints('Constraints/COIL20_constraints_onehalf_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, len(data), len(data))
# gcl.store_constraints('Constraints/COIL20_constraints_2n.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_2.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_3.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_4.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_5.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_6.txt', ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_5.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_6.txt')
# print Metrics.consistency(target, ml, cl)

# data, target = dataSetPreprocessing.loadIsolet()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.25*len(data), 0.25*len(data))
# gcl.store_constraints('Constraints/ISOLET_constraints_half_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.5*len(data), 0.5*len(data))
# gcl.store_constraints('Constraints/ISOLET_constraints_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.75*len(data), 0.75*len(data))
# gcl.store_constraints('Constraints/ISOLET_constraints_onehalf_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, len(data), len(data))
# gcl.store_constraints('Constraints/ISOLET_constraints_2n.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 760, 760)
# gcl.store_constraints('Constraints/ISOLET_constraints_1520_2.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_3.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_4.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_5.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 720, 720)
# gcl.store_constraints('Constraints/COIL20_constraints_1440_6.txt', ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_5.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_1440_6.txt')
# print Metrics.consistency(target, ml, cl)

# data, target = dataSetPreprocessing.loadIsolet()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 250, 250)
# gcl.store_constraints('Constraints/ISOLET_constraints_500_2.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 250, 250)
# gcl.store_constraints('Constraints/ISOLET_constraints_500_3.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 250, 250)
# gcl.store_constraints('Constraints/ISOLET_constraints_500_4.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 250, 250)
# gcl.store_constraints('Constraints/ISOLET_constraints_500_5.txt', ml, cl)

# ml, cl = gcl.read_constraints('Constraints/ISOLET_constraints_500_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/ISOLET_constraints_500_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/ISOLET_constraints_500_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/ISOLET_constraints_500_5.txt')
# print Metrics.consistency(target, ml, cl)


# data, target = ld.load_digit()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.25*len(data), 0.25*len(data))
# gcl.store_constraints('Constraints/OptDigits_constraints_half_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.5*len(data), 0.5*len(data))
# gcl.store_constraints('Constraints/OptDigits_constraints_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.75*len(data), 0.75*len(data))
# gcl.store_constraints('Constraints/OptDigits_constraints_onehalf_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, len(data), len(data))
# gcl.store_constraints('Constraints/OptDigits_constraints_2n.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 500, 500)
# gcl.store_constraints('Constraints/OptDigits_constraints_1000_1.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 500, 500)
# gcl.store_constraints('Constraints/OptDigits_constraints_1000_2.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 500, 500)
# gcl.store_constraints('Constraints/OptDigits_constraints_1000_3.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 500, 500)
# gcl.store_constraints('Constraints/OptDigits_constraints_1000_4.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 500, 500)
# gcl.store_constraints('Constraints/OptDigits_constraints_1000_5.txt', ml, cl)

# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_1000_1.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_1000_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_1000_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_1000_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_1000_5.txt')
# print Metrics.consistency(target, ml, cl)

# data, target = ld.load_segmentation()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.25*len(data), 0.25*len(data))
# gcl.store_constraints('Constraints/segmentation_constraints_half_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.5*len(data), 0.5*len(data))
# gcl.store_constraints('Constraints/segmentation_constraints_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 0.75*len(data), 0.75*len(data))
# gcl.store_constraints('Constraints/segmentation_constraints_onehalf_n.txt', ml, cl)
#
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, len(data), len(data))
# gcl.store_constraints('Constraints/segmentation_constraints_2n.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 350, 350)
# gcl.store_constraints('Constraints/segmentation_constraints_700_1.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 350, 350)
# gcl.store_constraints('Constraints/segmentation_constraints_700_2.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 350, 350)
# gcl.store_constraints('Constraints/segmentation_constraints_700_3.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 350, 350)
# gcl.store_constraints('Constraints/segmentation_constraints_700_4.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 350, 350)
# gcl.store_constraints('Constraints/segmentation_constraints_700_5.txt', ml, cl)

# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_700_1.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_700_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_700_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_700_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_700_5.txt')
# print Metrics.consistency(target, ml, cl)

data, target = ld.load_wdbc()
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 100, 100)
# gcl.store_constraints('Constraints/wdbc_constraints_200_1.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 100, 100)
# gcl.store_constraints('Constraints/wdbc_constraints_200_2.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 100, 100)
# gcl.store_constraints('Constraints/wdbc_constraints_200_3.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 100, 100)
# gcl.store_constraints('Constraints/wdbc_constraints_200_4.txt', ml, cl)
# ml,cl,_1,_2 = gcl.generate_closure_constraints_with_portion(target, 100, 100)
# gcl.store_constraints('Constraints/wdbc_constraints_200_5.txt', ml, cl)

# ml, cl = gcl.read_constraints('Constraints/wdbc_constraints_200_1.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/wdbc_constraints_200_2.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/wdbc_constraints_200_3.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/wdbc_constraints_200_4.txt')
# print Metrics.consistency(target, ml, cl)
# ml, cl = gcl.read_constraints('Constraints/wdbc_constraints_200_5.txt')
# print Metrics.consistency(target, ml, cl)

"""
do-stat
"""
# Metrics.do_performance_stat_in_folder('Results/COIL20/', 'stat.csv')
# Metrics.do_performance_stat_in_folder('Results/MNIST4000/', 'stat.csv')

"""
Selection Ensemble
"""
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500.txt')
# se.consistency_selection_ensemble_for_library('Results/COIL20/', 'COIL20_20-200_0.6_0.6_160_FSRSNC',
#                                               mlset, nlset, lm.get_default_logger(), 0.1, 0.5)
# se.k_selection_ensemble_for_library('Results/COIL20/', 'COIL20_20-200_0.6_0.6_160_FSRSNC', 90, lm.get_default_logger())
# se.k_selection_ensemble_for_library('Results/COIL20/', 'COIL20_20-200_0.6_0.6_160_FSRSNC', 100, lm.get_default_logger())
# se.k_selection_ensemble_for_library('Results/COIL20/', 'COIL20_20-200_0.6_0.6_160_FSRSNC', 110, lm.get_default_logger())


"""
MNIST E2CP
"""
# import constrained_clustering as cc
# data, target = dataSetPreprocessing.load_mnist_4000()
# print data.shape
# data = data.astype(float)
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_1000.txt')
# t1 = time.clock()
# e2cp = cc.E2CP(data=data, ml=ml, cl=cl, n_clusters=10)
# t2 = time.clock()
# e2cp.fit_constrained()
# print t2 - t1

"""
COIL20 E2CP
"""
# import constrained_clustering as cc
# data, target = dataSetPreprocessing.load_coil20()
# print data.shape
# data = data.astype(float)
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_500.txt')
# e2cp = cc.E2CP(data=data, ml=ml, cl=cl, n_clusters=20)
# t1 = time.clock()
# e2cp.fit_constrained()
# t2 = time.clock()
# print Metrics.normalized_max_mutual_info_score(target, e2cp.labels)
# print t2 - t1

"""
kNN, ANN tests
"""
#
# def build_hypergraph_adjacency(cluster_runs):
#     """Return the adjacency matrix to a hypergraph, in sparse matrix representation.
#
#     Parameters
#     ----------
#     cluster_runs : array of shape (n_partitions, n_samples)
#
#     Returns
#     -------
#     hypergraph_adjacency : compressed sparse row matrix
#         Represents the hypergraph associated with an ensemble of partitions,
#         each partition corresponding to a row of the array 'cluster_runs'
#         provided at input.
#     """
#
#     N_runs = cluster_runs.shape[0]
#
#     hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
#     for i in xrange(1, N_runs):
#         hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
#                                                     create_membership_matrix(cluster_runs[i])],
#                                                    format='csr')
#
#     return hypergraph_adjacency
# import operator
#
# def create_membership_matrix(cluster_run):
#     """For a label vector represented by cluster_run, constructs the binary
#         membership indicator matrix. Such matrices, when concatenated, contribute
#         to the adjacency matrix for a hypergraph representation of an
#         ensemble of clusterings.
#
#     Parameters
#     ----------
#     cluster_run : array of shape (n_partitions, n_samples)
#
#     Returns
#     -------
#     An adjacnecy matrix in compressed sparse row form.
#     """
#
#     cluster_run = np.asanyarray(cluster_run)
#
#     if reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
#         raise ValueError("\nERROR: Cluster_Ensembles: create_membership_matrix: "
#                          "problem in dimensions of the cluster label vector "
#                          "under consideration.")
#     else:
#         cluster_run = cluster_run.reshape(cluster_run.size)
#
#         cluster_ids = np.unique(np.compress(np.isfinite(cluster_run), cluster_run))
#
#         indices = np.empty(0, dtype=np.int32)
#         indptr = np.zeros(1, dtype=np.int32)
#
#         for elt in cluster_ids:
#             indices = np.append(indices, np.where(cluster_run == elt)[0])
#             indptr = np.append(indptr, indices.size)
#
#         data = np.ones(indices.size, dtype=int)
#
#         return scipy.sparse.csr_matrix((data, indices, indptr), shape=(cluster_ids.size, cluster_run.size))
#
# if __name__ == '__main__':
#     print 'Starting....'
#     data = np.loadtxt('covtype.data', delimiter=',')
#     print data.shape
#     targets = data[:, -1].flatten() - 1
#     print np.unique(targets)
#     print targets.shape
#     data = data[:, :-1]
#     print data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     X_train_minmax = min_max_scaler.fit_transform(data)
#     labels = []
#     for i in range(0, 160):
#         km = cluster.KMeans(n_clusters=7, n_init=1)
#         km.fit(X_train_minmax)
#         print str(i+1)+' clustering completed.'
#         labels.append(km.labels_)
#     t1 = time.clock()
#     hypergraph = build_hypergraph_adjacency(np.array(labels, dtype=np.int32))
#     hypergraph = hypergraph.transpose().tocsr()
#     t2 = time.clock()
#     print 'build matrix time = '+str(t2 - t1)
#     print hypergraph.shape
#     print hypergraph.data.size
#     del labels
#     gc.collect()
#     from sklearn.neighbors import NearestNeighbors
#     from sklearn.neighbors import LSHForest
#     from sklearn.neighbors import kneighbors_graph
#     t1 = time.clock()
#     # lshf = LSHForest(n_estimators=10, n_candidates=100, n_neighbors=50)
#     # lshf.fit(hypergraph)
#     # nbrs = NearestNeighbors(n_neighbors=50, n_jobs=4).fit(hypergraph)
#     conn = kneighbors_graph(hypergraph, n_neighbors=20)
#     t2 = time.clock()
#     print 'build knn time='+str(t2 - t1)
#
#     # gc.collect()
#     # t1 = time.clock()
#     # distances, indices = lshf.kneighbors(hypergraph[0])
#     # t2 = time.clock()
#     # print 'find neighbors time = '+str(t2 - t1)
#     #
#     # t1 = time.clock()
#     # distances, indices = lshf.kneighbors(hypergraph)
#     # t2 = time.clock()
#     # print 'find neighbors time2 = '+str(t2 - t1)
#     # s = scipy.sparse.csr_matrix.dot(hypergraph.transpose().tocsr(), hypergraph)
#     # print s.shape
#     # print s.data.size
#     # print Metrics.normalized_max_mutual_info_score(targets, km.labels_)
#
#     # print labels.shape

# logger = lm.get_default_logger()
# logger.debug('**********************************************************************************************')
# logger.debug('**********************************************************************************************')
# logger.debug('**********************************************************************************************')
# se.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.9_0.6_160_FSRSNC',
#                                     'Constraints/MNIST4000_constraints_4000_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='both')
# se.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.9_0.6_160_FSRSNC',
#                                     'Constraints/MNIST4000_constraints_4000_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='must')
# se.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.9_0.6_160_FSRSNC',
#                                     'Constraints/MNIST4000_constraints_4000_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='cannot')
# se.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.9_0.6_160_FSRSNC',
#                                     'Constraints/MNIST4000_constraints_4000_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='avg')

# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_1440_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='both')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_1440_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='must')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_1440_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='cannot')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_1440_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='avg')

# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_500.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='both')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_500.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='must')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_500.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='cannot')
# se.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.5_0.2_160_FSRSNC',
#                                     'Constraints/COIL20_constraints_500.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='avg')

# se.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_26-260_0.5_0.3_160_FSRSNC',
#                                     'Constraints/ISOLET_constraints_1520_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='both')
# se.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_26-260_0.5_0.3_160_FSRSNC',
#                                     'Constraints/N_constraints.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='must')
# se.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_26-260_0.5_0.3_160_FSRSNC',
#                                     'Constraints/N_constraints.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='cannot')
# se.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_26-260_0.5_0.3_160_FSRSNC',
#                                     'Constraints/ISOLET_constraints_1520_2.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='avg')

# mnist_threshold_tuples = [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]
# se.batch_do_consistency_selection_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.9_0.6_160_FSRSNC',
#                                               'Constraints/MNIST4000_constraints_1000.txt', lm.get_default_logger(),
#                                               mnist_threshold_tuples)

# se.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_26-260_0.5_0.3_160_FSRSNC',
#                                     'Constraints/N_Constraints.txt', lm.get_default_logger(),
#                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                     cons_type='both')


# data, target = dataSetPreprocessing.load_mnist_4000()
# data, target = dataSetPreprocessing.load_coil20()
# data, target = dataSetPreprocessing.loadIsolet()
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_4000_2.txt')
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000.txt')
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_1440_2.txt')
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500.txt')
# mlset, nlset = gcl.read_constraints('Constraints/N_constraints.txt')
# mlset, nlset = gcl.read_constraints('Constraints/ISOLET_constraints_1520_2.txt')
# labels = eck.cop_kmeans_wrapper(data, 20, mlset, nlset)
# print Metrics.normalized_max_mutual_info_score(target, labels)
# e2cp = cc.E2CP(data=data, ml=mlset, cl=nlset, n_clusters=20)
# e2cp.fit_constrained()
# labels = e2cp.labels
# print Metrics.normalized_max_mutual_info_score(target, labels)
# print Metrics.consistency_per_cluster(target, mlset, nlset, cons_type='both')


"""
check k-consistency correlation
"""
import utils.cluster_visualization as cvvv

# labels = np.loadtxt('Results/ISOLET/ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure.res', delimiter=',')
# mlset, nlset = gcl.read_constraints('Constraints/ISOLET_constraints_500_2.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_must_2.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_cannot_2.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/ISOLET_constraints_500_3.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_must_3.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_cannot_3.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/ISOLET_constraints_500_4.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_must_4.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_cannot_4.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/ISOLET_constraints_500_5.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_must_5.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'ISOLET_cannot_5.png', cons_type='cannot')


# labels = np.loadtxt('Results/MNIST4000/MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure.res', delimiter=',')
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000_2.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_must_1.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_cannot_1.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000_3.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_must_2.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_cannot_2.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000_4.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_must_3.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_cannot_3.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000_5.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_must_4.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_cannot_4.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/MNIST4000_constraints_1000_6.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_must_5.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'MNIST_cannot_5.png', cons_type='cannot')


# labels = np.loadtxt('Results/COIL20/COIL20_100-200_0.8_0.4_1000_FSRSNC_pure.res', delimiter=',')
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500_2.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_must_1.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_cannot_1.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500_3.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_must_2.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_cannot_2.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500_4.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_must_3.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_cannot_3.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500_5.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_must_4.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_cannot_4.png', cons_type='cannot')
#
# mlset, nlset = gcl.read_constraints('Constraints/COIL20_constraints_500_6.txt')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_must_5.png', cons_type='must')
# cvvv.plot_k_consistency_distribution(labels, mlset, nlset, 'COIL20_cannot_5.png', cons_type='cannot')

import ensemble.weighted_ensemble as weighted
#
# logger = lm.get_default_logger()
mnist_d, mnist_t = ld.load_mnist_4000()
isolet_d, isolet_t = ld.loadIsolet()
digit_d, digit_t = ld.load_digit()
seg_d, seg_t = ld.load_segmentation()
wdbc_d, wdbc_t = ld.load_wdbc()
coil_d, coil_t = ld.load_coil20()

# t1 = time.clock()
# a = metrics.silhouette_score(coil_d, coil_t, metric='euclidean')
# t2 = time.clock()
# print a
# print t2 - t1

#
# logger.debug('**********************************************************************************************')
# logger.debug('**********************************************************************************************')
# logger.debug('**********************************************************************************************')
#
# weighted.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure', 10, mnist_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='both', scale=True)
# weighted.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure', 10, mnist_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='must', scale=True)
#
#
# weighted.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.8_0.4_1000_FSRSNC_pure', 20, coil_t,
#                                           'Constraints/COIL20_constraints_500_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='both', scale=True)
# weighted.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.8_0.4_1000_FSRSNC_pure', 20, coil_t,
#                                           'Constraints/COIL20_constraints_500_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='must', scale=True)
#
#
# weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='both', scale=True)
# weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='must', scale=True)

# a = weighted.do_weighted_ensemble_for_library('Results/OPTDIGITS/', 'OPTDIGITS_50-100_0.8_0.8_1000_FSRSNC_pure', 10, digit_t,
#                                               'Constraints/OptDigits_constraints_1000_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
# b = weighted.do_weighted_ensemble_for_library('Results/OPTDIGITS/', 'OPTDIGITS_50-100_0.8_0.8_1000_FSRSNC_pure', 10, digit_t,
#                                               'Constraints/OptDigits_constraints_1000_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
#
# c = weighted.do_weighted_ensemble_for_library('Results/OPTDIGITS/', 'OPTDIGITS_50-100_0.8_0.8_1000_FSRSNC_pure', 10, digit_t,
#                                               'Constraints/OptDigits_constraints_1000_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=False)
# d = weighted.do_weighted_ensemble_for_library('Results/OPTDIGITS/', 'OPTDIGITS_50-100_0.8_0.8_1000_FSRSNC_pure', 10, digit_t,
#                                               'Constraints/OptDigits_constraints_1000_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=False)
# try:
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     d = np.array(d)
#     all_perf = np.hstack([a, b, c, d])
#     np.savetxt('OptDigits_weighted.csv', all_perf, fmt="%.6f", delimiter=',')
# except Exception, e:
#     print str(e)
#
# a = weighted.do_weighted_ensemble_for_library('Results/WDBC/', 'WDBC_10-20_0.7_0.5_1000_FSRSNC_pure', 2, wdbc_t,
#                                               'Constraints/wdbc_constraints_200_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
# b = weighted.do_weighted_ensemble_for_library('Results/WDBC/', 'WDBC_10-20_0.7_0.5_1000_FSRSNC_pure', 2, wdbc_t,
#                                               'Constraints/wdbc_constraints_200_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
#
# c = weighted.do_weighted_ensemble_for_library('Results/WDBC/', 'WDBC_10-20_0.7_0.5_1000_FSRSNC_pure', 2, wdbc_t,
#                                               'Constraints/wdbc_constraints_200_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=False)
# d = weighted.do_weighted_ensemble_for_library('Results/WDBC/', 'WDBC_10-20_0.7_0.5_1000_FSRSNC_pure', 2, wdbc_t,
#                                               'Constraints/wdbc_constraints_200_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=False)
# try:
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     d = np.array(d)
#     all_perf = np.hstack([a, b, c, d])
#     np.savetxt('WDBC_weighted.csv', all_perf, fmt="%.6f", delimiter=',')
# except Exception, e:
#     print str(e)
#
# a = weighted.do_weighted_ensemble_for_library('Results/segmentation/', 'segmentation_35-70_0.8_0.9_1000_FSRSNC_pure', 7, seg_t,
#                                               'Constraints/segmentation_constraints_700_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
# b = weighted.do_weighted_ensemble_for_library('Results/segmentation/', 'segmentation_35-70_0.8_0.9_1000_FSRSNC_pure', 7, seg_t,
#                                               'Constraints/segmentation_constraints_700_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
#
# c = weighted.do_weighted_ensemble_for_library('Results/segmentation/', 'segmentation_35-70_0.8_0.9_1000_FSRSNC_pure', 7, seg_t,
#                                               'Constraints/segmentation_constraints_700_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=False)
# d = weighted.do_weighted_ensemble_for_library('Results/segmentation/', 'segmentation_35-70_0.8_0.9_1000_FSRSNC_pure', 7, seg_t,
#                                               'Constraints/segmentation_constraints_700_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=False)
# try:
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     d = np.array(d)
#     all_perf = np.hstack([a, b, c, d])
#     np.savetxt('segmentation_weighted.csv', all_perf, fmt="%.6f", delimiter=',')
# except Exception, e:
#     print str(e)
#
#
# a = weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                               'Constraints/ISOLET_constraints_500_2.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=True)
# b = weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                               'Constraints/ISOLET_constraints_500_2.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=True)
#
# c = weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                               'Constraints/ISOLET_constraints_500_2.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='both', scale=False)
# d = weighted.do_weighted_ensemble_for_library('Results/ISOLET/', 'ISOLET_130-260_0.7_0.5_1000_FSRSNC_pure', 26, isolet_t,
#                                               'Constraints/ISOLET_constraints_500_2.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=False)
#
# try:
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     d = np.array(d)
#     all_perf = np.hstack([a, b, c, d])
#     np.savetxt('ISOLET_weighted.csv', all_perf, fmt="%.6f", delimiter=',')
# except Exception, e:
#     print str(e)

# b = weighted.do_weighted_ensemble_for_library('Results/OPTDIGITS/', 'OPTDIGITS_50-100_0.8_0.8_1000_FSRSNC_pure', 10, digit_t,
#                                               'Constraints/OptDigits_constraints_1000_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=True)
# np.savetxt('OptDigits_nn_weighted.csv', b, fmt="%.6f", delimiter=',')
#
# b = weighted.do_weighted_ensemble_for_library('Results/WDBC/', 'WDBC_10-20_0.7_0.5_1000_FSRSNC_pure', 2, wdbc_t,
#                                               'Constraints/wdbc_constraints_200_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=True)
# np.savetxt('WDBC_nn_weighted.csv', b, fmt="%.6f", delimiter=',')
#
# b = weighted.do_weighted_ensemble_for_library('Results/segmentation/', 'segmentation_35-70_0.8_0.9_1000_FSRSNC_pure', 7, seg_t,
#                                               'Constraints/segmentation_constraints_700_1.txt', lm.get_default_logger(),
#                                               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                               cons_type='must', scale=True)
# np.savetxt('segmentation_nn_weighted.csv', b, fmt="%.6f", delimiter=',')

# a = weighted.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure', 10, mnist_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='both', scale=False)
# b = weighted.do_weighted_ensemble_for_library('Results/MNIST4000/', 'MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure', 10, mnist_t,
#                                           'Constraints/MNIST4000_constraints_1000_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='must', scale=False)
# a = np.array(a)
# b = np.array(b)
# all_perf = np.hstack([a, b])
# np.savetxt('MNIST_nn_weighted.csv', all_perf, fmt="%.6f", delimiter=',')
#
# a = weighted.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.8_0.4_1000_FSRSNC_pure', 20, coil_t,
#                                           'Constraints/COIL20_constraints_500_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='both', scale=False)
# b = weighted.do_weighted_ensemble_for_library('Results/COIL20/', 'COIL20_100-200_0.8_0.4_1000_FSRSNC_pure', 20, coil_t,
#                                           'Constraints/COIL20_constraints_500_2.txt', lm.get_default_logger(),
#                                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                                           cons_type='must', scale=False)
# a = np.array(a)
# b = np.array(b)
# all_perf = np.hstack([a, b])
# np.savetxt('COIL_nn_weighted.csv', all_perf, fmt="%.6f", delimiter=',')

"""
check segmentation
"""
# ml, cl = gcl.read_constraints('Constraints/segmentation_constraints_2n.txt')

# km = cluster.KMeans(n_clusters=7)
# km.fit(seg_d)
# label = km.labels_

# e2cp = cc.E2CP(data=seg_d, ml=ml, cl=cl, n_clusters=7)
# e2cp.fit_constrained()
# label = e2cp.labels

# label = mk.cop_kmeans_wrapper(seg_d, 7, ml, cl)

# labels = np.loadtxt('Results/segmentation/segmentation_35-70_0.7_1.0_1000_Cop_KMeans_segmentation_constraints_700_1.txt_pure.res', delimiter=',')
# label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=7)

# label = spec.spectral_ensemble(labels, N_clusters_max=7)

# print Metrics.normalized_max_mutual_info_score(seg_t, label)

"""
check optdigits
"""
# ml, cl = gcl.read_constraints('Constraints/OptDigits_constraints_2n.txt')

# km = cluster.KMeans(n_clusters=10)
# km.fit(digit_d)
# label = km.labels_
#
# e2cp = cc.E2CP(data=digit_d, ml=ml, cl=cl, n_clusters=10)
# e2cp.fit_constrained()
# label = e2cp.labels

# label = mk.cop_kmeans_wrapper(digit_d, 10, ml, cl)
#
# labels = np.loadtxt('Results/OPTDIGITS/OPTDIGITS_50-100_0.7_1.0_1000_Cop_KMeans_OptDigits_constraints_1000_1.txt_pure.res', delimiter=',')
# # label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=10)
# #
# label = spec.spectral_ensemble(labels, N_clusters_max=10)
# #
# print Metrics.normalized_max_mutual_info_score(digit_t, label)

"""
check isolet
"""
# ml, cl = gcl.read_constraints('Constraints/ISOLET_constraints_2n.txt')

# km = cluster.KMeans(n_clusters=10)
# km.fit(digit_d)
# label = km.labels_

# e2cp = cc.E2CP(data=isolet_d, ml=ml, cl=cl, n_clusters=26)
# e2cp.fit_constrained()
# label = e2cp.labels

# label = mk.cop_kmeans_wrapper(isolet_d, 26, ml, cl)

# labels = np.loadtxt('Results/ISOLET/ISOLET_130-260_0.7_1.0_1000_Cop_KMeans_ISOLET_constraints_500_2.txt_pure.res', delimiter=',')
#
# label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=26)
#
# label = spec.spectral_ensemble(labels, N_clusters_max=26)
#
# print Metrics.normalized_max_mutual_info_score(isolet_t, label)

"""
check coil20
"""
# ml, cl = gcl.read_constraints('Constraints/COIL20_constraints_2n.txt')

# km = cluster.KMeans(n_clusters=10)
# km.fit(digit_d)
# label = km.labels_

# e2cp = cc.E2CP(data=coil_d, ml=ml, cl=cl, n_clusters=20)
# e2cp.fit_constrained()
# label = e2cp.labels

# label = mk.cop_kmeans_wrapper(coil_d, 20, ml, cl)

# labels = np.loadtxt('Results/COIL20/COIL20_100-200_0.7_1.0_1000_Cop_KMeans_COIL20_constraints_500.txt_pure.res', delimiter=',')
#
# label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=20)

# label = spec.spectral_ensemble(labels, N_clusters_max=20)
#
# print Metrics.normalized_max_mutual_info_score(coil_t, label)


"""
check mnist4000
"""
# ml, cl = gcl.read_constraints('Constraints/MNIST4000_constraints_2n.txt')

# km = cluster.KMeans(n_clusters=10)
# km.fit(digit_d)
# label = km.labels_

# e2cp = cc.E2CP(data=mnist_d, ml=ml, cl=cl, n_clusters=10)
# e2cp.fit_constrained()
# label = e2cp.labels

# label = mk.cop_kmeans_wrapper(mnist_d, 10, ml, cl)

# labels = np.loadtxt('Results/MNIST4000/MNIST4000_50-100_0.7_1.0_1000_Cop_KMeans_MNIST4000_constraints_1000.txt_pure.res', delimiter=',')
#
# label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=10)
#
# label = spec.spectral_ensemble(labels, N_clusters_max=10)
#
# print Metrics.normalized_max_mutual_info_score(mnist_t, label)

# aa = {'a':1, 'b':2}
# for i in aa:
#     print aa[i]
# from ensemble.Cluster_Ensembles import build_hypergraph_adjacency
# a = np.loadtxt('Results/MNIST4000/MNIST4000_50-100_0.6_0.9_1000_FSRSNC_pure.res', delimiter=',')
# adjc = build_hypergraph_adjacency(a)
# adjc = adjc.transpose()
#
# att_mat = adjc.dot(adjc.transpose())
# att_mat = np.squeeze(np.asarray(att_mat.todense()))
# np.savetxt('testA.txt', att_mat, fmt='%d', delimiter=',')

import constrained_methods.generate_constraints_link as gcl
print gcl._build_default_amount_array(100)
