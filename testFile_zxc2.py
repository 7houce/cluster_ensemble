import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
import constrained_clustering as cc
import dataSetPreprocessing as dp
import generate_constraints_link as gcl


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
np.savetxt('Results/kmean_e2cp_constrast.txt', result)

