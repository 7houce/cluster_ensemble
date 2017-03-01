import dataSetPreprocessing as dp
import ensemble_generation as eg
import math
import time
# ensemble about isolet using e2cp and efficient_cop_kmeans

ISOLET_dataSets = {'ISOLET': dp.loadIsolet}


ISO_con_params_Cop_1 = {'ISOLET': {'members': 160, 'classNum': 26, 'small_Clusters': 26, 'large_Clusters': 260,
                              'method': 'Cop_KMeans', 'constraints': 'Constraints/N_constraints.txt'}}


ISO_con_params_E2CP_1 = {'ISOLET': {'members': 160, 'classNum': 26, 'small_Clusters': 26, 'large_Clusters': 260,
                              'method': 'E2CP', 'constraints': 'Constraints/N_constraints.txt'}}




eg.autoGenerationWithConsensus(ISOLET_dataSets, ISO_con_params_Cop_1, metric='NID', manifold_type='MDS', subfolder=True)

eg.autoGenerationWithConsensus(ISOLET_dataSets, ISO_con_params_E2CP_1, metric='NID', manifold_type='MDS', subfolder=True)


