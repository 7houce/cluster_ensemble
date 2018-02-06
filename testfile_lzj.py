import constrained_methods.constrained_clustering as cc
import utils.load_dataset as ld
import utils.io_func as io
import time
import evaluation.Metrics as Metrics

data, target = ld.load_mnist_4000()
print data.shape
data = data.astype(float)
ml, cl = io.read_constraints('Constraints/MNIST4000_diff_n_1.txt')
t1 = time.clock()
e2cp = cc.E2CP(data=data, ml=ml, cl=cl, n_clusters=10)
t2 = time.clock()
e2cp.fit_constrained()
print e2cp.labels
print Metrics.normalized_max_mutual_info_score(target, e2cp.labels)
print t2 - t1
