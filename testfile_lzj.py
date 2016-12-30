import basicClusterMethods as bcm
import dataSetPreprocessing as dSP
import Cluster_Ensembles as ce
import numpy as np
import sys
import Metrics
from sklearn import cluster

# put the dataset-loading functions in a dictionary
datasets = {'digit': dSP.loadDigits}
name = 'digit'
data, target = datasets[name]()

# experiment parameters
tag = True
verbose = True
classnum = 10
A = classnum
B = 10 * classnum
ISR = 0.7
FSR = 0.7
n_members = 20

# matrix to store clustering results
mat = np.empty(data.shape[0])

# generate ensemble members
for i in range(0, n_members):
    cluster_num = np.random.randint(A, B)
    random_state = np.random.randint(0, sys.maxint - 1)
    result = bcm.FSRSNN_c(data, target, r_clusters=cluster_num, r_state=random_state)
    # print diversity
    diver = Metrics.diversityBtw2Cluster(result, target)
    if verbose:
        print 'This time diversity = ' + str(diver)
    # stack the result into the matrix
    if tag:
        mat = np.array(result)
        mat = np.reshape(mat, (1, data.shape[0]))
        tag = False
    else:
        temp = np.array(result)
        temp = np.reshape(temp, (1, data.shape[0]))
        mat = np.vstack([mat, np.array(temp)])

mat = mat.astype(int)

# consensus
labels_CSPA = ce.cluster_ensembles_CSPAONLY(mat, N_clusters_max=classnum)
labels_HGPA = ce.cluster_ensembles_HGPAONLY(mat, N_clusters_max=classnum)
labels_MCLA = ce.cluster_ensembles_MCLAONLY(mat, N_clusters_max=classnum)
print labels_CSPA
print labels_HGPA
print labels_MCLA

# put consensus results into the matrix
mat = np.vstack([mat, np.reshape(labels_CSPA, (1, data.shape[0]))])
mat = np.vstack([mat, np.reshape(labels_HGPA, (1, data.shape[0]))])
mat = np.vstack([mat, np.reshape(labels_MCLA, (1, data.shape[0]))])

# path and filename to write the file
path = 'Results/'
fileName = name + '_' + str(A) + '-' + str(B) + '_' + str(ISR) + '_' + str(FSR) + '_' + str(n_members) + '.res'
print 'Results are saving to file : ' + fileName

# write results to external file, use %d to keep integer part only
np.savetxt(path + fileName, mat, fmt='%d', delimiter=',')

clf = cluster.KMeans(n_clusters=10)
clf.fit(data)
kmlabels = clf.labels_

# print labels
nmi_CSPA = Metrics.diversityBtw2Cluster(labels_CSPA, target)
nmi_HGPA = Metrics.diversityBtw2Cluster(labels_HGPA, target)
nmi_MCLA = Metrics.diversityBtw2Cluster(labels_MCLA, target)
print 'final diversity CSPA ='+str(nmi_CSPA)
print 'final diversity HGPA ='+str(nmi_HGPA)
print 'final diversity MCLA ='+str(nmi_MCLA)

kmnmi = Metrics.diversityBtw2Cluster(kmlabels, target)
print 'final diversity k-means ='+str(kmnmi)
