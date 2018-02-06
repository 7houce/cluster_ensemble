import utils.exp_datasets as ed
import evaluation.Metrics as metrics
# from sklearn import metrics
import constrained_methods.constrained_clustering as cc
import constrained_methods.efficient_cop_kmeans as eck
import utils.io_func as io
import numpy as np
import pandas as pd
from sklearn import cluster
import ensemble.Cluster_Ensembles as ce
import ensemble.spectral_ensemble as spec
import time

# from sklearn.cross_validation import train_test_split
#
#
# def training(data, type=1):
#     if type != 1:
#         data = data.as_matrix()
#         data = np.asarray(data)
#     result = []
#     for ds in data:
#         result.append(np.average(ds))
#     return np.array(result)
#
# d = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]])
# target = np.array([1, 6, 4, 7, 3])
#
# number = np.array(range(1, d.shape[0]+1))
# data_selected, data_unselected, \
# target_selected, target_unselected = train_test_split(d, number,
#                                                       train_size=3,
#                                                       random_state=222)
# print data_selected
# print data_unselected
# print target_selected
# print target_unselected
#
# pred_selected = training(data_selected)
# pred_unselected = training(data_unselected)
#
# reunion_numbers = np.hstack([target_selected, target_unselected])
# predicted_labels = np.hstack([pred_selected, pred_unselected])
# print predicted_labels[np.argsort(reunion_numbers)]

# dd = pd.DataFrame(d)
# tt = pd.DataFrame(target)
# print type(dd)
# data_selected, data_unselected, \
# target_selected, target_unselected = train_test_split(dd, tt,
#                                                       train_size=3,
#                                                       random_state=222)
# pred_selected = training(data_selected, type=2)
# pred_unselected = training(data_unselected, type=2)
#
# result_selected = target_selected.copy()
# result_selected[1] = np.array(pred_selected)
#
# result_unselected = target_unselected.copy()
# result_unselected[1] = np.array(pred_unselected)
#
# result = pd.concat([result_selected, result_unselected])
# result = result.reindex(range(0, len(target)))

# print result[1].values

# print '--------------------'
# for doo in data_selected.as_matrix():
#     print doo

d, t = ed.dataset['waveform']['data']()
# d, t = ed.dataset['Wap']['data'](sparse_type='csr')
# d, t = ed.dataset['k1b']['data']()
# d, t = ed.dataset['hitech']['data']()
# d, t = ed.dataset['re0']['data']()
print d.shape
print np.unique(t)
km = cluster.KMeans(n_clusters=3)
t1 = time.clock()
km.fit(d)
t2 = time.clock()
print metrics.normalized_max_mutual_info_score(t, km.labels_)
# metrics
print t2-t1
# import member_generation.subspace as sub
# subd = sub.feature_sampling(d, 2000)
# print d.shape
# print subd.shape
# data_selected, data_unselected, \
# target_selected, target_unselected = train_test_split(d, t,
#                                                       train_size=500,
#                                                       random_state=154)
# print data_selected
# print data_unselected
# print target_selected
# print target_unselected
# print d
# ml, cl = io.read_constraints('Constraints/Wap_constraints_2n.txt')
# ml, cl = io.read_constraints('Constraints/k1b_constraints_2n.txt')
ml, cl = io.read_constraints('Constraints/waveform_constraints_half_n.txt')
print metrics.consistency(t, ml, cl)
# e2cp = cc.E2CP(data=d, ml=ml, cl=cl, n_clusters=6)
# t1 = time.clock()
# e2cp.fit_constrained()
# t2 = time.clock()
# print t
# print np.unique(t)
# print metrics.normalized_max_mutual_info_score(t, e2cp.labels)
# print (t2 - t1)
# t1 = time.clock()
label = eck.cop_kmeans_wrapper(d, 3, ml, cl)
# t2 = time.clock()
# km = cluster.KMeans(n_clusters=20)
# km.fit(d)
print metrics.normalized_max_mutual_info_score(t, label)
# print (t2 - t1)
# labels = [[1,1,1,2,2,3,3],
#           [2,2,2,3,3,1,1],
#           [1,1,2,2,3,3,3]]
# labels = np.array(labels)
# hyperedges = ce.build_hypergraph_adjacency(labels)
# print hyperedges.transpose().toarray()
# coas_matrix = np.dot(hyperedges.transpose().toarray(), hyperedges.transpose().toarray().transpose())
# coas_matrix = coas_matrix.astype(np.float32)
# coas_matrix /= np.max(coas_matrix)
# print coas_matrix
# label = ce.cluster_ensembles_CSPA_on_matrix(coas_matrix, 3, 2)
# label = spec.spectral_ensemble_on_matrix(coas_matrix, 3, 2)
# print label
# print labels
# label = ce.cluster_ensembles_CSPAONLY(labels, N_clusters_max=2)
# print label
