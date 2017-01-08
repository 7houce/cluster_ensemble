import dataSetPreprocessing as dSP
import ensemble_generation as eg

dataSets = {'digit': dSP.loadDigits, 'movement': dSP.loadMovement_libras, 'Synthetic': dSP.loadSynthetic_control}
paramSettings1 = {'digit': {'members': 160, 'classNum': 10, 'FSR': 0.7, 'SSR': 0.5},
                  'movement': {'members': 160, 'classNum': 15, 'FSR': 0.7, 'SSR': 0.5},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.7, 'SSR': 0.5}
                 }
paramSettings2 = {'digit': {'members': 160, 'classNum': 10, 'FSR': 0.3, 'SSR': 0.3},
                  'movement': {'members': 160, 'classNum': 15, 'FSR': 0.3, 'SSR': 0.3},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.3, 'SSR': 0.3}
                 }
paramSettings3 = {'digit': {'members': 160, 'classNum': 10, 'FSR': 0.5, 'SSR': 0.5},
                  'movement': {'members': 160, 'classNum': 15, 'FSR': 0.5, 'SSR': 0.5},
                  'Synthetic': {'members': 160, 'classNum': 6, 'FSR': 0.5, 'SSR': 0.5}
                 }

eg.autoGenerationWithConsensus(dataSets, paramSettings1)
eg.autoGenerationWithConsensus(dataSets, paramSettings2)
eg.autoGenerationWithConsensus(dataSets, paramSettings3)


