import dataSetPreprocessing as dSP
import ensemble_generation as eg
import math

dataSets = {'Digit': dSP.loadDigits, 'Movement': dSP.loadMovement_libras, 'Synthetic': dSP.loadSynthetic_control,
            'Glass': dSP.loadGlass, 'Ionosphere': dSP.loadIonosphere, 'Iris': dSP.loadIris,
            'Wine': dSP.loadWine}

ISOdatasets = {'ISOLET': dSP.loadIsolet}

irisdatasets = {'Iris': dSP.loadIris}

irisparamsettings = {'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.7, 'SSR': 0.7}}

ISOparamSettings1 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.05, 'SSR': 0.7}}
ISOparamSettings2 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3, 'SSR': 0.3}}
ISOparamSettings3 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3, 'SSR': 0.7}}
ISOparamSettings4 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': math.sqrt(1559), 'FSR_L': 5, 'SSR': 0.7}}
ISOparamSettings5 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': math.sqrt(1559), 'FSR_L': 5, 'SSR': 0.3}}

gisettedataset = {'gisette': dSP.load_gisette_data}

gisetteparamsettings = {'gisette': {'members': 160, 'classNum': 2, 'FSR': 0.05, 'SSR': 0.5}}

paramSettings1 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.7, 'SSR': 0.5},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.7, 'SSR': 0.5},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.7, 'SSR': 0.5},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.7, 'SSR': 0.5},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.7, 'SSR': 0.5},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.7, 'SSR': 0.5},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.7, 'SSR': 0.5},
                 }

paramSettings2 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.3, 'SSR': 0.3},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.3, 'SSR': 0.3},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.3, 'SSR': 0.3},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.3, 'SSR': 0.3},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.3, 'SSR': 0.3},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.3, 'SSR': 0.3},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.3, 'SSR': 0.3},
                 }

paramSettings3 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.5, 'SSR': 0.5},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.5, 'SSR': 0.5},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.5, 'SSR': 0.5},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.5, 'SSR': 0.5},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.5, 'SSR': 0.5},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.5, 'SSR': 0.5},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.5, 'SSR': 0.5},
                 }

paramSettings4 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.7, 'SSR': 0.7},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.7, 'SSR': 0.7},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.7, 'SSR': 0.7},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.7, 'SSR': 0.7},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.7, 'SSR': 0.7},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.7, 'SSR': 0.7},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.7, 'SSR': 0.7},
                 }

paramSettings5 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.1, 'SSR': 0.1},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.1, 'SSR': 0.1},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.1, 'SSR': 0.1},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.1, 'SSR': 0.1},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.1, 'SSR': 0.1},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.1, 'SSR': 0.1},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.1, 'SSR': 0.1},
                 }

paramSettings6 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.9, 'SSR': 0.9},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.9, 'SSR': 0.9},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.9, 'SSR': 0.9},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.9, 'SSR': 0.9},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.9, 'SSR': 0.9},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.9, 'SSR': 0.9},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.9, 'SSR': 0.9},
                 }

paramSettings7 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.8, 'SSR': 0.2},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.8, 'SSR': 0.2},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.8, 'SSR': 0.2},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.8, 'SSR': 0.2},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.8, 'SSR': 0.2},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.8, 'SSR': 0.2},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.8, 'SSR': 0.2},
                 }

paramSettings8 = {'Digit': {'members': 160, 'classNum': 10, 'FSR': 0.2, 'SSR': 0.8},
                  'Movement': {'members': 160, 'classNum': 15, 'FSR': 0.2, 'SSR': 0.8},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.2, 'SSR': 0.8},
                  'Glass': {'members': 160, 'classNum': 7, 'FSR': 0.2, 'SSR': 0.8},
                  'Ionosphere': {'members': 160, 'classNum': 2, 'FSR': 0.2, 'SSR': 0.8},
                  'Iris': {'members': 160, 'classNum': 3, 'FSR': 0.2, 'SSR': 0.8},
                  'Wine': {'members': 160, 'classNum': 3, 'FSR': 0.2, 'SSR': 0.8},
                 }


ISO_de_params_1 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3, 'small_Clusters': 16, 'large_Clusters': 36,
                              'SSR': 0.5, 'method': 'FSRSNC', 'constraints': 'Constraints/N_constraints.txt'}}

ISO_de_params_2 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3, 'small_Clusters': 26, 'large_Clusters': 46,
                              'SSR': 0.5, 'method': 'FSRSNC', 'constraints': 'Constraints/N_constraints.txt'}}

ISO_de_params_3 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3, 'small_Clusters': 26, 'large_Clusters': 26,
                              'SSR': 0.5, 'method': 'FSRSNC', 'constraints': 'Constraints/N_constraints.txt'}}

ISO_de_params_3 = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.3,
                              'SSR': 0.5, 'method': 'FSRSNC', 'constraints': 'Constraints/N_constraints.txt'}}

eg.autoGenerationWithConsensus(ISOdatasets, ISO_de_params_1, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(ISOdatasets, ISO_de_params_2, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(ISOdatasets, ISO_de_params_3, metric='NID', manifold_type='MDS', subfolder=True)


# FSRs = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
# SSRs = [0.5, 0.7, 0.9]
# for i in range(0, 10):
#     for j in range(0, 3):
#         ISO_de_params['ISOLET']['FSR'] = FSRs[i]
#         ISO_de_params['ISOLET']['SSR'] = SSRs[j]
#         eg.autoGenerationWithConsensus(ISOdatasets, ISO_de_params, metric='NID', manifold_type='MDS', subfolder=True)
#



# eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings1, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings2, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings3, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings4, metric='NID', manifold_type='MDS', subfolder=True, stable_sample=False)
# eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings5, metric='NID', manifold_type='MDS', subfolder=True, stable_sample=False)

# eg.autoGenerationWithConsensus(gisettedataset, gisetteparamsettings, metric='NID', manifold_type='MDS', subfolder=True)

# eg.autoGenerationWithConsensus(irisdatasets, irisparamsettings, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings1, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings2, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings3, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings4, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings5, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings6, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings7, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings8, metric='NID', manifold_type='MDS', subfolder=True)
