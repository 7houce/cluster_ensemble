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
import os

colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod', 'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']

def spin(D):
    n = D.shape[0]
    w = np.array(range(1, n + 1), dtype=float)
    w = w.T - float(n)/2
    w = np.reshape(w, (n, 1))
    print w.shape
    P_prev = np.eye(n)
    P_prev = P_prev.astype(float)
    stop = False
    iter = 0
    while not stop:
        iter += 1
        s = np.dot(D, P_prev.T)
        s = np.dot(s, w)
        p = np.argsort(-s, axis=0)

        P = coo_matrix((np.array([1]*n), (np.squeeze(np.array(range(0, n))), np.squeeze(p))))
        P = P.toarray()

        cur_norm_term = np.dot(P, s)
        prev_norm_term = np.dot(P_prev, s)
        norm = LA.norm(cur_norm_term - prev_norm_term, np.inf)
        if norm < 1e-7:
            stop = True
        P_prev = P
        print P
        i, j = P.nonzero()
    return i, j


def rotateMatrix(D, index):
    n = index.shape[0]
    total = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            row.append(D[index[i]][index[j]])
        total.append(row)
    return np.array(total)

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
SPIN method
"""
# diverMatrix = np.loadtxt('Results/Synthetic/Synthetic_6-60_0.7_0.7_160_nid.txt', delimiter=',')
# plt.imshow(diverMatrix, vmin=0, vmax=1)
# plt.show()
# i, j = spin(diverMatrix)
# print i
# print j
# newMatrix = rotateMatrix(diverMatrix, j)
# print newMatrix
# print newMatrix.shape
# print np.max(newMatrix)
# print np.min(newMatrix)
# plt.imshow(diverMatrix, vmin=np.min(diverMatrix), vmax=np.max(diverMatrix))
# plt.show()
# plt.imshow(newMatrix, vmin=np.min(newMatrix), vmax=np.max(newMatrix))
# plt.show()
#


def drawOrderedDistanceMatrix(path, subfolder='odm'):
    path = os.path.expanduser(path)
    for f in os.listdir(path):
        fullpath = os.path.join(path, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            distanceMatrix = np.loadtxt(fullpath, delimiter=',')
            plt.imshow(distanceMatrix, vmin=np.min(distanceMatrix), vmax=np.max(distanceMatrix))
            plt.savefig(path + subfolder + '/' + fname[0] + '_original.svg', format='svg', dpi=240)
            plt.clf()
            i, j = spin(distanceMatrix)
            newMatrix = rotateMatrix(distanceMatrix, j)
            plt.imshow(newMatrix, vmin=np.min(newMatrix), vmax=np.max(newMatrix))
            plt.savefig(path + subfolder + '/' + fname[0] + '_after.svg', format='svg', dpi=240)

# drawOrderedDistanceMatrix('Results/distance/')
# traverseDirByListdir('Results/')
# traverseDirByOSWalk('Results/')
