"""
dataset used for experiment
"""
import utils.load_dataset as ld

dataset = {
    'OPTDIGITS': {'k': 10, 'data': ld.load_digit},
    'Segmentation': {'k': 7, 'data': ld.load_segmentation},
    'ISOLET': {'k': 26, 'data': ld.loadIsolet},
    'MNIST4000': {'k': 10, 'data': ld.load_mnist_4000},
    'COIL20': {'k': 20, 'data': ld.load_coil20},
    'DLBCL': {'k': 2, 'data': ld.load_DLBCL},
    'Prostate': {'k': 2, 'data': ld.load_Prostate_Tumor},
    'SRBCT': {'k': 4, 'data': ld.load_SRBCT},
    'LungCancer': {'k': 4, 'data': ld.load_Lung_Cancer},
    'Leukemia1': {'k': 3, 'data': ld.load_Leukemia1},
    'Leukemia2': {'k': 3, 'data': ld.load_Leukemia2},
    'Iris': {'k': 3, 'data': ld.load_iris},
    'Wap': {'k': 20, 'data': ld.load_wap},
    'k1b': {'k': 6, 'data': ld.load_k1b},
    'hitech': {'k': 6, 'data': ld.load_hitech},
    're0': {'k': 13, 'data': ld.load_re0},
    're1': {'k': 25, 'data': ld.load_re1},
    'waveform': {'k': 3, 'data': ld.load_waveform}
}
