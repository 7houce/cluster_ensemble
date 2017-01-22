import numpy as np
import Metrics
import time
import dataSetPreprocessing
import matplotlib.pyplot as plt
from MSTClustering import MSTClustering
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import datetime
import pickle
from sklearn import cluster
from sklearn import metrics
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import random as rand
import spin
import os

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


# drawOrderedDistanceMatrix('Results/distance/')
# traverseDirByListdir('Results/')
# traverseDirByOSWalk('Results/')


# test code for the generation
# target = [1,1,1,1,2,2,2,3,3,1,1,4,4,4,2,2,3,3]
# m, c = generateconstraints(target, n=5)
#
# for i, j in m:
#     print str(i) + ',' + str(j)
# print '=============================='
# for i, j in c:
#     print str(i) + ',' + str(j)
