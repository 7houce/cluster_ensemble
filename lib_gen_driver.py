import member_generation.library_generation as lg
import utils.io_func as io_func
import utils.exp_datasets as exd
import evaluation.Metrics as metrics
_postfix = ['diff_n_1', 'diff_n_2', 'diff_n_3', 'diff_n_4', 'diff_n_5']
_noise_postfix = ['noise_n_1', 'noise_n_2', 'noise_n_3', 'noise_n_4', 'noise_n_5']
#
# lg.generate_libs_by_sampling_rate('Iris', 100)
# lg.generate_libs_by_sampling_rate('OPTDIGITS', 100)
# lg.generate_libs_by_sampling_rate('Segmentation', 100)
# lg.generate_libs_by_sampling_rate('ISOLET', 100)
# lg.generate_libs_by_sampling_rate('MNIST4000', 100)
# lg.generate_libs_by_sampling_rate('COIL20', 100)
# lg.generate_libs_by_sampling_rate('Prostate', 100)
# lg.generate_libs_by_sampling_rate('SRBCT', 100)
# lg.generate_libs_by_sampling_rate('LungCancer', 100)
# lg.generate_libs_by_sampling_rate('Leukemia1', 100)
# lg.generate_libs_by_sampling_rate('Leukemia2', 100)

# import constrained_methods.generate_constraints_link as gcl
# gcl.generate_diff_amount_constraints_wrapper('OPTDIGITS')
# gcl.generate_diff_amount_constraints_wrapper('Segmentation')
# gcl.generate_diff_amount_constraints_wrapper('ISOLET')
# gcl.generate_diff_amount_constraints_wrapper('MNIST4000')
# gcl.generate_diff_amount_constraints_wrapper('COIL20')
# gcl.generate_diff_amount_constraints_wrapper('Prostate')
# gcl.generate_diff_amount_constraints_wrapper('SRBCT')
# gcl.generate_diff_amount_constraints_wrapper('LungCancer')
# gcl.generate_diff_amount_constraints_wrapper('Leukemia2')

import evaluation.comparision_methods as ed
import member_generation.library_generation as lg
import ensemble.ensemble_wrapper as ew
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False)

# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, additional_postfix='_informative')
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, additional_postfix='_informative')

"""
weight ensemble (different constraints in same volume)
"""
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)

#
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_postfix)

# ed.comparison_methods('OPTDIGITS')
# ed.comparison_methods('Segmentation')
# ed.comparison_methods('ISOLET')
# ed.comparison_methods('MNIST4000')
# ed.comparison_methods('COIL20')
# ed.comparison_methods('Prostate')
# ed.comparison_methods('SRBCT')
# ed.comparison_methods('LungCancer')
# ed.comparison_methods('Leukemia2')
# ed.comparison_methods('Leukemia1')

# ed.comparison_methods('OPTDIGITS', additional_postfix='_informative')
# ed.comparison_methods('Segmentation', additional_postfix='_informative')
# ed.comparison_methods('ISOLET', additional_postfix='_informative')
# ed.comparison_methods('MNIST4000', additional_postfix='_informative')
# ed.comparison_methods('COIL20', additional_postfix='_informative')
# ed.comparison_methods('Prostate', additional_postfix='_informative')
# ed.comparison_methods('SRBCT', additional_postfix='_informative')
# ed.comparison_methods('LungCancer', additional_postfix='_informative')
# ed.comparison_methods('Leukemia2', additional_postfix='_informative')
# ed.comparison_methods('Leukemia1', additional_postfix='_informative')
#
# ed.comparison_methods('OPTDIGITS', constraints_files=_postfix)
# ed.comparison_methods('Segmentation', constraints_files=_postfix)
# ed.comparison_methods('ISOLET', constraints_files=_postfix)
# ed.comparison_methods('MNIST4000', constraints_files=_postfix)
# ed.comparison_methods('COIL20', constraints_files=_postfix)
# ed.comparison_methods('Prostate', constraints_files=_postfix)
# ed.comparison_methods('SRBCT', constraints_files=_postfix)
# ed.comparison_methods('LungCancer', constraints_files=_postfix)
# ed.comparison_methods('Leukemia2', constraints_files=_postfix)
# ed.comparison_methods('Leukemia1', constraints_files=_postfix)
#
# ed.comparison_methods('OPTDIGITS', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('Segmentation', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('ISOLET', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('MNIST4000', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('COIL20', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('Prostate', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('SRBCT', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('LungCancer', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('Leukemia2', additional_postfix='_informative', constraints_files=_postfix)
# ed.comparison_methods('Leukemia1', additional_postfix='_informative', constraints_files=_postfix)

"""
amount e2cp
"""
# lg.generate_libs_by_constraints('OPTDIGITS', 100)
# lg.generate_libs_by_constraints('Segmentation', 100)
# lg.generate_libs_by_constraints('ISOLET', 100)
# lg.generate_libs_by_constraints('MNIST4000', 100)
# lg.generate_libs_by_constraints('COIL20', 100)
# lg.generate_libs_by_constraints('Prostate', 100)
# lg.generate_libs_by_constraints('SRBCT', 100)
# lg.generate_libs_by_constraints('LungCancer', 100)
# lg.generate_libs_by_constraints('Leukemia2', 100)
# lg.generate_libs_by_constraints('Leukemia1', 100)

"""
amount e2cp informative
"""
# lg.generate_libs_by_constraints('Prostate', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('SRBCT', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('LungCancer', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia2', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia1', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('OPTDIGITS', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Segmentation', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('ISOLET', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('MNIST4000', 100, additional_postfix='_informative')
# lg.generate_libs_by_constraints('COIL20', 100, additional_postfix='_informative')


"""
amount informative cop
"""
# lg.generate_libs_by_constraints('OPTDIGITS', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Segmentation', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('ISOLET', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('MNIST4000', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('COIL20', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Prostate', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('SRBCT', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('LungCancer', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia2', 100, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia1', 100, member_method='Cop_KMeans', additional_postfix='_informative')

"""
amount cop
"""
# lg.generate_libs_by_constraints('OPTDIGITS', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Segmentation', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('ISOLET', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('MNIST4000', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('COIL20', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Prostate', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('SRBCT', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('LungCancer', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Leukemia2', 100, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Leukemia1', 100, member_method='Cop_KMeans')


"""
diff_n informative cop
"""
# lg.generate_libs_by_constraints('OPTDIGITS', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('Segmentation', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('ISOLET', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('MNIST4000', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('COIL20', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('Prostate', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('SRBCT', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('LungCancer', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('Leukemia2', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)
# lg.generate_libs_by_constraints('Leukemia1', 100, member_method='Cop_KMeans', additional_postfix='_informative', postfixes=_postfix)

"""
diff_n random cop
"""
# lg.generate_libs_by_constraints('OPTDIGITS', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('Segmentation', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('ISOLET', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('MNIST4000', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('COIL20', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('Prostate', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('SRBCT', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('LungCancer', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('Leukemia2', 100, member_method='Cop_KMeans', postfixes=_postfix)
# lg.generate_libs_by_constraints('Leukemia1', 100, member_method='Cop_KMeans', postfixes=_postfix)

"""
diff_n informative e2cp
"""
# lg.generate_libs_by_constraints('Prostate', 100, postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('SRBCT', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('LungCancer', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia2', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia1', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('OPTDIGITS', 100, postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Segmentation', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('ISOLET', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('MNIST4000', 100,  postfixes=_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('COIL20', 100,  postfixes=_postfix, additional_postfix='_informative')



import utils.informative_constraints_selection as ics
"""
Generate informative cannot-link set.
"""
# ics.generate_informative_cl_set('OPTDIGITS', n_nbor=30)
# ics.generate_informative_cl_set('Segmentation', n_nbor=20)
# ics.generate_informative_cl_set('ISOLET', n_nbor=20)
# ics.generate_informative_cl_set('MNIST4000', n_nbor=20)
# ics.generate_informative_cl_set('COIL20', n_nbor=20)
# ics.generate_informative_cl_set('Prostate')
# ics.generate_informative_cl_set('SRBCT')
# ics.generate_informative_cl_set('LungCancer', n_nbor=12)
# ics.generate_informative_cl_set('Leukemia2')
# ics.generate_informative_cl_set('Leukemia1')

"""
Generate constraints with different amount (informative)
"""
# gcl.generate_diff_amount_constraints_wrapper('OPTDIGITS', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('Segmentation', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('ISOLET', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('MNIST4000', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('COIL20', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('Prostate', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('SRBCT', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('LungCancer', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('Leukemia2', informative=True)
# gcl.generate_diff_amount_constraints_wrapper('Leukemia1', informative=True)

"""
Generate different constraints in the same amount
"""
# gcl.generate_diff_constraints_wrapper('OPTDIGITS')
# gcl.generate_diff_constraints_wrapper('Segmentation')
# gcl.generate_diff_constraints_wrapper('ISOLET')
# gcl.generate_diff_constraints_wrapper('MNIST4000')
# gcl.generate_diff_constraints_wrapper('COIL20')
# gcl.generate_diff_constraints_wrapper('Prostate')
# gcl.generate_diff_constraints_wrapper('SRBCT')
# gcl.generate_diff_constraints_wrapper('LungCancer')
# gcl.generate_diff_constraints_wrapper('Leukemia2')
# gcl.generate_diff_constraints_wrapper('Leukemia1')
#
# gcl.generate_diff_constraints_wrapper('OPTDIGITS', informative=True)
# gcl.generate_diff_constraints_wrapper('Segmentation', informative=True)
# gcl.generate_diff_constraints_wrapper('ISOLET', informative=True)
# gcl.generate_diff_constraints_wrapper('MNIST4000', informative=True)
# gcl.generate_diff_constraints_wrapper('COIL20', informative=True)
# gcl.generate_diff_constraints_wrapper('Prostate', informative=True)
# gcl.generate_diff_constraints_wrapper('SRBCT', informative=True)
# gcl.generate_diff_constraints_wrapper('LungCancer', informative=True)
# gcl.generate_diff_constraints_wrapper('Leukemia2', informative=True)
# gcl.generate_diff_constraints_wrapper('Leukemia1', informative=True)

# gcl.generate_noise_constraints_wrapper('OPTDIGITS')
# gcl.generate_noise_constraints_wrapper('Segmentation')
# gcl.generate_noise_constraints_wrapper('ISOLET')
# gcl.generate_noise_constraints_wrapper('MNIST4000')
# gcl.generate_noise_constraints_wrapper('COIL20')
# gcl.generate_noise_constraints_wrapper('Prostate')
# gcl.generate_noise_constraints_wrapper('SRBCT')
# gcl.generate_noise_constraints_wrapper('LungCancer')
# gcl.generate_noise_constraints_wrapper('Leukemia2')
# gcl.generate_noise_constraints_wrapper('Leukemia1')

# gcl.generate_noise_constraints_wrapper('OPTDIGITS', informative=True)
# gcl.generate_noise_constraints_wrapper('Segmentation', informative=True)
# gcl.generate_noise_constraints_wrapper('ISOLET', informative=True)
# gcl.generate_noise_constraints_wrapper('MNIST4000', informative=True)
# gcl.generate_noise_constraints_wrapper('COIL20', informative=True)
# gcl.generate_noise_constraints_wrapper('Prostate', informative=True)
# gcl.generate_noise_constraints_wrapper('SRBCT', informative=True)
# gcl.generate_noise_constraints_wrapper('LungCancer', informative=True)
# gcl.generate_noise_constraints_wrapper('Leukemia2', informative=True)
# gcl.generate_noise_constraints_wrapper('Leukemia1', informative=True)


# dataset_name = 'Segmentation'
# constraint_name = '_noise_n_2.txt'
# ml, cl = io_func.read_constraints('Constraints/' + dataset_name + constraint_name)
# d, t = exd.dataset[dataset_name]['data']()
# print metrics.consistency(t, ml, cl)

"""
weight ensemble (noise constraints in same volume)
"""
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
#
#
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, additional_postfix='_informative', constraints_files_postfix=_noise_postfix)

"""
constrained methods ensemble (noise same volume)
"""
# lg.generate_libs_by_constraints('Prostate', 100, postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('SRBCT', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('LungCancer', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Leukemia2', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Leukemia1', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('OPTDIGITS', 100, postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('Segmentation', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('ISOLET', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('MNIST4000', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
# lg.generate_libs_by_constraints('COIL20', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans')
#
# lg.generate_libs_by_constraints('Prostate', 100, postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('SRBCT', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('LungCancer', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('Leukemia2', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('Leukemia1', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('OPTDIGITS', 100, postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('Segmentation', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('ISOLET', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('MNIST4000', 100,  postfixes=_noise_postfix)
# lg.generate_libs_by_constraints('COIL20', 100,  postfixes=_noise_postfix)

# lg.generate_libs_by_constraints('Prostate', 100, postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('SRBCT', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('LungCancer', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia2', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia1', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('OPTDIGITS', 100, postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('Segmentation', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('ISOLET', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('MNIST4000', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
# lg.generate_libs_by_constraints('COIL20', 100,  postfixes=_noise_postfix, member_method='Cop_KMeans', additional_postfix='_informative')
#
# lg.generate_libs_by_constraints('Prostate', 100, postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('SRBCT', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('LungCancer', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia2', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Leukemia1', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('OPTDIGITS', 100, postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('Segmentation', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('ISOLET', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('MNIST4000', 100,  postfixes=_noise_postfix, additional_postfix='_informative')
# lg.generate_libs_by_constraints('COIL20', 100,  postfixes=_noise_postfix, additional_postfix='_informative')

"""
comparison methods for noise constraints.
"""
# ed.comparison_methods('OPTDIGITS', constraints_files=_noise_postfix)
# ed.comparison_methods('Segmentation', constraints_files=_noise_postfix)
# ed.comparison_methods('ISOLET', constraints_files=_noise_postfix)
# ed.comparison_methods('MNIST4000', constraints_files=_noise_postfix)
# ed.comparison_methods('COIL20', constraints_files=_noise_postfix)
# ed.comparison_methods('Prostate', constraints_files=_noise_postfix)
# ed.comparison_methods('SRBCT', constraints_files=_noise_postfix)
# ed.comparison_methods('LungCancer', constraints_files=_noise_postfix)
# ed.comparison_methods('Leukemia2', constraints_files=_noise_postfix)
# ed.comparison_methods('Leukemia1', constraints_files=_noise_postfix)
#
# ed.comparison_methods('OPTDIGITS', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('Segmentation', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('ISOLET', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('MNIST4000', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('COIL20', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('Prostate', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('SRBCT', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('LungCancer', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('Leukemia2', additional_postfix='_informative', constraints_files=_noise_postfix)
# ed.comparison_methods('Leukemia1', additional_postfix='_informative', constraints_files=_noise_postfix)

"""
===========================================================================================================
new experiments, 12th Dec, 2017.
updates: new formula for calculating weights (g_gamma introduced)
12.27 modified: internals included
===========================================================================================================
"""
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False)

# result
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, constraints_files_postfix=_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, constraints_files_postfix=_postfix)
#
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Segmentation_35-70_0.5_0.8_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.9_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('ISOLET_130-260_0.6_0.4_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('SRBCT_20-40_0.6_0.6_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Prostate_10-20_0.8_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('LungCancer_20-40_0.9_0.3_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
# ew.do_ensemble_different_constraints_new_exp('Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure', False, constraints_files_postfix=_noise_postfix)

import evaluation.internal_metrics as im



import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster.unsupervised import _intra_cluster_distance
from sklearn.metrics.cluster.unsupervised import _nearest_cluster_distance
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
from numpy import bincount
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as metricss

# def check_number_of_labels(n_labels, n_samples):
#     if not 1 < n_labels < n_samples:
#         raise ValueError("Number of labels is %d. Valid values are 2 "
#                          "to n_samples - 1 (inclusive)" % n_labels)


# def silhouette_samples(X, labels, metric='euclidean', **kwds):
#
#     X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)
#     check_number_of_labels(len(le.classes_), X.shape[0])
#
#     tttt1 = time.clock()
#     distances = pairwise_distances(X, metric=metric, **kwds)
#     tttt2 = time.clock()
#     print 'time is '+str(tttt2-tttt1)
#     unique_labels = le.classes_
#     n_samples_per_label = bincount(labels, minlength=len(unique_labels))
#
#     # For sample i, store the mean distance of the cluster to which
#     # it belongs in intra_clust_dists[i]
#     intra_clust_dists = np.zeros(distances.shape[0], dtype=distances.dtype)
#
#     # For sample i, store the mean distance of the second closest
#     # cluster in inter_clust_dists[i]
#     inter_clust_dists = np.inf + intra_clust_dists
#
#     for curr_label in range(len(unique_labels)):
#
#         # Find inter_clust_dist for all samples belonging to the same
#         # label.
#         mask = labels == curr_label
#         current_distances = distances[mask]
#
#         # Leave out current sample.
#         n_samples_curr_lab = n_samples_per_label[curr_label] - 1
#         if n_samples_curr_lab != 0:
#             intra_clust_dists[mask] = np.sum(
#                 current_distances[:, mask], axis=1) / n_samples_curr_lab
#
#         # Now iterate over all other labels, finding the mean
#         # cluster distance that is closest to every sample.
#         for other_label in range(len(unique_labels)):
#             if other_label != curr_label:
#                 other_mask = labels == other_label
#                 other_distances = np.mean(
#                     current_distances[:, other_mask], axis=1)
#                 inter_clust_dists[mask] = np.minimum(
#                     inter_clust_dists[mask], other_distances)
#
#     sil_samples = inter_clust_dists - intra_clust_dists
#     sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
#     # score 0 for clusters of size 1, according to the paper
#     sil_samples[n_samples_per_label.take(labels) == 1] = 0
#     return sil_samples

# # import utils.exp_datasets as ed
# d, t = exd.dataset['MNIST4000']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure')
# d, t = exd.dataset['OPTDIGITS']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure')
# d, t = exd.dataset['Segmentation']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'Segmentation_35-70_0.5_0.8_100_FSRSNC_pure')
# d, t = exd.dataset['COIL20']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'COIL20_100-200_0.9_0.2_100_FSRSNC_pure')
# d, t = exd.dataset['ISOLET']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'ISOLET_130-260_0.6_0.4_100_FSRSNC_pure')
# d, t = exd.dataset['SRBCT']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'SRBCT_20-40_0.6_0.6_100_FSRSNC_pure')
# d, t = exd.dataset['Prostate']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'Prostate_10-20_0.8_0.3_100_FSRSNC_pure')
# d, t = exd.dataset['LungCancer']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'LungCancer_20-40_0.9_0.3_100_FSRSNC_pure')
# d, t = exd.dataset['Leukemia1']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure')
# d, t = exd.dataset['Leukemia2']['data']()
# ret_v = im.cal_internal_weights_for_library_cluster_as_array(d, 'Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure')
# for v in ret_v:
#     print v
# library_path = 'Results/'
# dataset_name = 'ISOLET'
# library_name = 'ISOLET_130-260_0.6_0.4_100_FSRSNC_pure'
# labels = np.loadtxt(library_path + dataset_name + '/' + library_name + '.res', delimiter=',')
# l = ce.cluster_ensembles_MCLAONLY(labels)
# print metricss.normalized_max_mutual_info_score(t, l)
# t1 = time.clock()
# silhouette_samples(d, labels[0])
# t2 = time.clock()
# print 'time='+str(t2-t1)
