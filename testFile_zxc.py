import basicClusterMethods as bcm
import dataSetPreprocessing as dp
import ensemble_generation as eg
import generate_constraints_link as gcl
import time
import Metrics as metric
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
import constrained_clustering as cc



# run single cop_KMeans in dataSet isolet-5 100 times

# dataSet, target = dp.loadIsolet()
# must_link, cannot_link = gcl.read_constraints('Constraints/N_constraints.txt')
# result = []
# for i in range(100):
#    row = []
#    t1 = time.clock()
#    clusters, centers = ck.cop_KMeans(dataSet, 26, must_link, cannot_link)
#    t2 = time.clock()
#    row.append(int(26))
#    row.append(float(t2-t1))
#    row.append(float(metric.normalized_max_mutual_info_score(target, clusters)))
#    result.append(row)
# np.savetxt('Results/result.txt', result)
# print 'Finished'

# read result of cop_KMeans

# fr = open('Results/result.txt')
# dataSet = []
#
# for line in fr.readlines():
#     curLine = line.strip().split()
#     fltLine = map(float, curLine)
#     dataSet.append(fltLine)
# dataSet = pd.DataFrame(dataSet)
# print np.mean(dataSet[1])

# Constrast between kmeans and e2cp in dataSet isolet-5

dataSet, target = dp.loadIsolet()
must_link, cannot_link = gcl.read_constraints('Constraints/N_constraints.txt')
result = []

for i in range(10):
    single = []
    N_clusters = 26
    kmeans = KMeans(n_clusters=N_clusters)
    kmeans.fit(dataSet)
    baseline = nmi(target, kmeans.labels_)
    single.append(round(baseline,3))
    print round(baseline,3)

    # E2CP
    e2cp = cc.E2CP(data=dataSet,
                   ml=must_link,
                   cl=cannot_link,
                   n_clusters=N_clusters)
    e2cp.fit_constrained()
    e2cpLabels = e2cp.labels
    single.append(round(nmi(target, e2cpLabels),3))
    print round(nmi(target, e2cpLabels),3)
    result.append(single)
np.savetxt('Results/kmeans_e2cp_constrast.txt', result)