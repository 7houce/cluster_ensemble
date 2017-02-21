import basicClusterMethods as bcm
import dataSetPreprocessing as dp
import ensemble_generation as eg
import generate_constraints_link as gcl
import time
import Metrics as metric
import numpy as np
import pandas as pd

#dataSets = {'digit': dp.loadDigits, 'movement': dp.loadMovement_libras}
#dataSets = {'digit': dp.loadDigits, 'movement': dp.loadMovement_libras, 'robot_1': dp.loadRobotExecution_1,
#            'robot_2': dp.loadRobotExecution_2, 'robot_4': dp.loadRobotExecution_4, 'synthetic': dp.loadSynthetic_control}

#commonParm = {'members': 20, 'classNum': 10, 'small_Clusters': 20, 'large_Clusters': 200, 'FSR': 1, 'SSR': 0.7}

#paramSettings = {'digit': commonParm,
#                 'movement': commonParm}

#eg.autoGenerationWithConsensus(dataSets, paramSettings)

# dataSet, target = dp.loadIris()
# must_link, cannot_link, ml_graph, cl_graph = gcl.generate_closure_constraints(target, 10)
# print must_link
# print cannot_link
# print ml_graph
# print cl_graph

#dataSet, target = dp.loadIsolet()
#must_link, cannot_link = gcl.read_constraints('Constraints/N_constraints.txt')
#result = []
#for i in range(100):
#    row = []
#    t1 = time.clock()
#    clusters, centers = ck.cop_KMeans(dataSet, 26, must_link, cannot_link)
#    t2 = time.clock()
#    row.append(int(26))
#    row.append(float(t2-t1))
#    row.append(float(metric.normalized_max_mutual_info_score(target, clusters)))
#    result.append(row)
#np.savetxt('Results/result.txt', result)
#print 'Finished'

# fr = open('Results/result.txt')
# dataSet = []
#
# for line in fr.readlines():
#     curLine = line.strip().split()
#     fltLine = map(float, curLine)
#     dataSet.append(fltLine)
# dataSet = pd.DataFrame(dataSet)
# print np.mean(dataSet[1])

#dataSet, target = dp.loadIsolet()
#must_link, cannot_link = gcl.read_constraints('Constraints/N_constraints.txt')
#clusters, centers = ck.cop_KMeans(dataSet, 26, must_link, cannot_link)
#print metric.consistency(clusters, must_link, cannot_link)


# gcl.generate(200)
# links = np.load('isolet.npy').item()
#
# dataSet, target = dp.loadIsolet()
#
# t1 = time.clock()
# clf = ck.ConstrainedKMeans(26)
# clf.fit(dataSet, target, **links)
# t2 = time.clock()
# print t2 - t1
# print clf.labels_
#
# print metric.normalized_max_mutual_info_score(target, clf.labels_)