import basicClusterMethods as bCM
import dataSetPreprocessing as dataPre

dataSet, target = dataPre.loadRobotExecution_1()
print (bCM.KMeans_c(dataSet))