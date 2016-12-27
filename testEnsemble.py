import Cluster_Ensembles as ce
import numpy as np


samples = np.array([0, 0, 0, 1, 1, 2, 2])
samples = np.vstack([samples, np.array([1, 1, 1, 2, 2, 0, 0])])
samples = np.vstack([samples, np.array([0, 0, 1, 1, 2, 2, 2])])
samples = np.vstack([samples, np.array([0, 1, np.NAN, 0, 1, np.NAN, np.NAN])])

# labels1 = ce.cluster_ensembles_CSPAONLY(samples)
# labels2 = ce.cluster_ensembles_HGPAONLY(samples)
# labels3 = ce.cluster_ensembles_MCLAONLY(samples)
#
# print labels1
# print labels2
# print labels3

# print samples.shape

samples = np.random.randint(0, 5, size=(200, 100000))
label = ce.cluster_ensembles_HGPAONLY(samples)
print label

# ones = np.ones((4, 1))
#
# print (ones)