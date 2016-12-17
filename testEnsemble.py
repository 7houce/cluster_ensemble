import Cluster_Ensembles as ce
import numpy as np

samples = np.array([0, 0, 0, 1, 1, 2, 2])
samples = np.vstack([samples, np.array([1, 1, 1, 2, 2, 0, 0])])
samples = np.vstack([samples, np.array([0, 0, 1, 1, 2, 2, 2])])
samples = np.vstack([samples, np.array([0, 1, np.NAN, 0, 1, np.NAN, np.NAN])])

labels = ce.cluster_ensembles_CSPAONLY(samples)

print labels

