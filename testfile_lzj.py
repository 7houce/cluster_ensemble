import dataSetPreprocessing as dSP
import ensemble_generation as eg

dataSets = {'digit': dSP.loadDigits, 'movement': dSP.loadMovement_libras}
paramSettings = {'digit': {'members': 160, 'classnum': 10},
                 'movement': {'members': 160, 'classnum': 15}}

eg.autoGenerationWithConsensus(dataSets, paramSettings)

