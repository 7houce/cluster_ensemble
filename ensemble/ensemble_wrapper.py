"""
Wrapper function for executing several experiments together.
Author: Zhijie Lin
"""
import numpy as np
import ensemble.weighted_ensemble as weighted
import logger_module as lm
import utils.exp_datasets as exp_data
import evaluation.internal_metrics as im

_default_logger = lm.get_default_logger()
_default_constraints_folder = 'Constraints/'
_default_result_folder = 'Results/'
_default_constraints_postfix = ['constraints_quarter_n', 'constraints_half_n', 'constraints_n', 'constraints_onehalf_n',
                                'constraints_2n']
_default_ensemble_performance_folder = 'Results/Exp_Results/'
_default_cons_types = ['both', 'must']
_default_alpha_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def do_ensemble_different_constraints(library_name, scale, cons_types=None, constraints_files_postfix=None,
                                      precomputed_internals=None, additional_postfix=''):
    """
    do weighted ensemble for a library using different constraints

    :param library_name:
    :param scale:
    :param cons_types:
    :param constraints_files_postfix:
    :param precomputed_internals:
    :param additional_postfix:
    :return:
    """
    dataset_name = library_name.split('_')[0]
    library_folder = _default_result_folder + dataset_name + '/'
    class_num = exp_data.dataset[dataset_name]['k']
    d, t = exp_data.dataset[dataset_name]['data']()
    scale_tag = '_scaled' if scale else ''
    weighted_cons_types = _default_cons_types if cons_types is None else cons_types
    files_postfix = _default_constraints_postfix if constraints_files_postfix is None else constraints_files_postfix
    if precomputed_internals is not None:
        internals = precomputed_internals
    else:
        internals = im.cal_internal_weights_for_library_as_array(d, library_name)
    for postfix in files_postfix:
        performances = []
        for cons_type in weighted_cons_types:
            constraints_file_name = _default_constraints_folder + dataset_name + '_' + postfix + additional_postfix + '.txt'
            # it should be changed to a generalized version later
            performance = weighted.do_7th_weighted_ensemble_for_library(library_folder,
                                                                        library_name, class_num, t,
                                                                        constraints_file_name,
                                                                        _default_logger,
                                                                        _default_alpha_range,
                                                                        internals,
                                                                        cons_type=cons_type, scale=scale)
            performances.append(np.array(performance))
        all_perf = np.hstack(performances)
        np.savetxt(_default_ensemble_performance_folder + dataset_name + '_' + postfix + scale_tag + additional_postfix + '.csv', all_perf,
                   fmt='%.6f', delimiter=',')
    return
