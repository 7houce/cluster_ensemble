import member_generation.library_generation as lg
import utils.exp_datasets as ed
import evaluation.comparision_methods as ecm
import constrained_methods.generate_constraints_link as gcl
import evaluation.Metrics as metrics
import utils.io_func as io
import ensemble.ensemble_wrapper as ew

_postfix = ['diff_n_1', 'diff_n_2', 'diff_n_3', 'diff_n_4', 'diff_n_5']
_noise_postfix = ['noise_n_1', 'noise_n_2', 'noise_n_3', 'noise_n_4', 'noise_n_5']

"""
load dataset
"""
# d, t = ed.dataset['Wap']['data']()

"""
library generation
"""
# lg.generate_libs_by_sampling_rate('Wap', 100)
# lg.generate_libs_by_sampling_rate('k1b', 100)
# lg.generate_libs_by_sampling_rate('hitech', 100)
# lg.generate_libs_by_sampling_rate('re0', 100)
# lg.generate_libs_by_sampling_rate('waveform', 100)

"""
constraints generation
"""
# gcl.generate_diff_amount_constraints_wrapper('Wap')
# gcl.generate_diff_constraints_wrapper('Wap')
# gcl.generate_noise_constraints_wrapper('Wap')

# gcl.generate_diff_amount_constraints_wrapper('k1b')
# gcl.generate_diff_constraints_wrapper('k1b')
# gcl.generate_noise_constraints_wrapper('k1b')

# gcl.generate_diff_amount_constraints_wrapper('hitech')
# gcl.generate_diff_constraints_wrapper('hitech')
# gcl.generate_noise_constraints_wrapper('hitech')

# gcl.generate_diff_amount_constraints_wrapper('re0')
# gcl.generate_diff_constraints_wrapper('re0')
# gcl.generate_noise_constraints_wrapper('re0')

# gcl.generate_diff_amount_constraints_wrapper('waveform')
# gcl.generate_diff_constraints_wrapper('waveform')
# gcl.generate_noise_constraints_wrapper('waveform')

"""
constrained library generation
"""
# lg.generate_libs_by_constraints('Wap', 100)
# lg.generate_libs_by_constraints('Wap', 100, postfixes=_postfix)
# lg.generate_libs_by_constraints('Wap', 100, postfixes=_noise_postfix)
#
# lg.generate_libs_by_constraints('Wap', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Wap', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('Wap', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')

# lg.generate_libs_by_constraints('k1b', 100)
# lg.generate_libs_by_constraints('k1b', 100, postfixes=_postfix)
# lg.generate_libs_by_constraints('k1b', 100, postfixes=_noise_postfix)
#
# lg.generate_libs_by_constraints('k1b', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('k1b', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('k1b', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')

# lg.generate_libs_by_constraints('hitech', 100)
# lg.generate_libs_by_constraints('hitech', 100, postfixes=_postfix)
# lg.generate_libs_by_constraints('hitech', 100, postfixes=_noise_postfix)
#
# lg.generate_libs_by_constraints('hitech', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('hitech', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('hitech', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')

# lg.generate_libs_by_constraints('re0', 100)
# lg.generate_libs_by_constraints('re0', 100, postfixes=_postfix)
# lg.generate_libs_by_constraints('re0', 100, postfixes=_noise_postfix)
#
# lg.generate_libs_by_constraints('re0', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('re0', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('re0', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')

# lg.generate_libs_by_constraints('waveform', 100)
# lg.generate_libs_by_constraints('waveform', 100, postfixes=_postfix)
# lg.generate_libs_by_constraints('waveform', 100, postfixes=_noise_postfix)
#
# lg.generate_libs_by_constraints('waveform', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('waveform', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('waveform', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')

"""
comparison methods
"""
# ecm.comparison_methods('Wap')
# ecm.comparison_methods('Wap', constraints_files=_noise_postfix)
# ecm.comparison_methods('Wap', constraints_files=_postfix)


"""
check consistency
"""
# ml, cl = io.read_constraints('Constraints/Wap_noise_n_1.txt')
# print metrics.consistency(t, ml, cl)

"""
weighting approach
"""
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', False)

#
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
#
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('hitech_30-60_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('k1b_30-60_0.7_0.7_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)

"""
propagated approach
"""
# ew.do_prop_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure')
# ew.do_prop_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure')
#
# ew.do_prop_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', constraints_files_postfix=_postfix)
# ew.do_prop_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', constraints_files_postfix=_postfix)
#
# ew.do_prop_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('waveform_15-30_0.9_0.9_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('re0_65-130_0.5_0.2_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)
# ew.do_prop_ensemble_different_constraints('Wap_100-200_0.7_0.8_100_FSRSNC_pure', constraints_files_postfix=_noise_postfix)

"""
Comparison ensemble
"""
# ecm.comparison_ensemble_methods('MNIST4000', 'MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('OPTDIGITS', 'OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('Segmentation', 'Segmentation_35-70_0.5_0.8_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('COIL20', 'COIL20_100-200_0.9_0.2_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('ISOLET', 'ISOLET_130-260_0.6_0.4_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('SRBCT', 'SRBCT_20-40_0.6_0.6_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('Prostate', 'Prostate_10-20_0.8_0.3_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('LungCancer', 'LungCancer_20-40_0.9_0.3_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('Leukemia1', 'Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('Leukemia2', 'Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('waveform', 'waveform_15-30_0.9_0.9_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('re0', 're0_65-130_0.5_0.2_100_FSRSNC_pure')
# ecm.comparison_ensemble_methods('Wap', 'Wap_100-200_0.7_0.8_100_FSRSNC_pure')

"""
Comparison (constrained)
"""
ecm.comparison_methods('Wap')
ecm.comparison_methods('re0')
ecm.comparison_methods('waveform')

ecm.comparison_methods('Wap', constraints_files=_postfix)
ecm.comparison_methods('re0', constraints_files=_postfix)
ecm.comparison_methods('waveform', constraints_files=_postfix)

ecm.comparison_methods('Wap', constraints_files=_noise_postfix)
ecm.comparison_methods('re0', constraints_files=_noise_postfix)
ecm.comparison_methods('waveform', constraints_files=_noise_postfix)
