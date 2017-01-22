import dataSetPreprocessing as dSP
import ensemble_generation as eg
import os

dataSets = {'Digit': dSP.loadDigits, 'Movement': dSP.loadMovement_libras, 'Synthetic': dSP.loadSynthetic_control,
            'Glass': dSP.loadGlass, 'Ionosphere': dSP.loadIonosphere, 'Iris': dSP.loadIris,
            'Wine': dSP.loadWine}

ISOdatasets = {'ISOLET': dSP.loadIsolet}

ISOparamSettings = {'ISOLET': {'members': 160, 'classNum': 26, 'FSR': 0.7, 'SSR': 0.7}}

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

# if os.path.isdir('Results/MST/mst/Wine'):
#     print 'is'
# else:
#     os.mkdir('Results/MST/mst/Wine')


eg.autoGenerationWithConsensus(ISOdatasets, ISOparamSettings, metric='NID', manifold_type='MDS', subfolder=True)

# eg.autoGenerationWithConsensus(dataSets, paramSettings1, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings2, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings3, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings4, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings5, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings6, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings7, metric='NID', manifold_type='MDS', subfolder=True)
# eg.autoGenerationWithConsensus(dataSets, paramSettings8, metric='NID', manifold_type='MDS', subfolder=True)
