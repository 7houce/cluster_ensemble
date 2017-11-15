from utils import load_dataset as dSP
import math
import member_generation.library_generation as lg
import evaluation.eval_library as el


# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.9, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         lg.generate_library(coil_data, coil_target, coil_name, 160, coil_class_num,
#                             n_cluster_lower_bound=5 * coil_class_num,
#                             n_cluster_upper_bound=10 * coil_class_num,
#                             feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                             sampling_method='FSRSNC', constraints_file=coil_constraints_file)
#
# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.9, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         lg.generate_library(mnist_data, mnist_target, mnist_name, 160, mnist_class_num,
#                             n_cluster_lower_bound=5 * mnist_class_num,
#                             n_cluster_upper_bound=10 * mnist_class_num,
#                             feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                             sampling_method='FSRSNC', constraints_file=mnist_constraints_file)

# isolet_name = 'ISOLET'
# isolet_constraints_file = 'Constraints/ISOLET-Constraints_n_1.txt'
# isolet_data, isolet_target = dSP.load_mnist_4000()
# isolet_class_num = 26
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.9, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         lg.generate_library(isolet_data, isolet_target, isolet_name, 160, isolet_class_num,
#                             n_cluster_lower_bound=5 * isolet_class_num,
#                             n_cluster_upper_bound=10 * isolet_class_num,
#                             feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                             sampling_method='FSRSNC', constraints_file=isolet_constraints_file)

# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_4000_2.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# lg.generate_library(mnist_data, mnist_target, mnist_name, 160, mnist_class_num,
#                     n_cluster_lower_bound=mnist_class_num,
#                     n_cluster_upper_bound=mnist_class_num,
#                     sampling_method='E2CP', constraints_file=mnist_constraints_file)

"""
COIL20 160 E2CP and Cop-KMeans ensemble (constraints=n)
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_1440_2.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# lg.generate_library(coil_data, coil_target, coil_name, 160, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='E2CP', constraints_file=coil_constraints_file)
#
# lg.generate_library(coil_data, coil_target, coil_name, 160, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=coil_constraints_file)
#

"""
MNIST4000 160 E2CP ensemble (constraints=1000)
"""
# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000_2.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# lg.generate_library(mnist_data, mnist_target, mnist_name, 160, mnist_class_num,
#                     n_cluster_lower_bound=5 * mnist_class_num,
#                     n_cluster_upper_bound=10 * mnist_class_num,
#                     sampling_method='E2CP', constraints_file=mnist_constraints_file)

"""
MNIST4000 160 Cop-KMeans ensemble (constraints=1000)
"""
# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# lg.generate_library(mnist_data, mnist_target, mnist_name, 160, mnist_class_num,
#                     n_cluster_lower_bound=5 * mnist_class_num,
#                     n_cluster_upper_bound=10 * mnist_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=mnist_constraints_file)

"""
COIL20 160 Cop-KMeans ensemble (constraints=500)
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# lg.generate_library(coil_data, coil_target, coil_name, 1000, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=coil_constraints_file)

"""
COIL20 160 E2CP ensemble (constraints=500)
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# lg.generate_library(coil_data, coil_target, coil_name, 1000, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='E2CP', constraints_file=coil_constraints_file)

"""
ISOLET 160 E2CP ensemble (constraints=500)
"""
# isolet_name = 'ISOLET'
# isolet_constraints_file = 'Constraints/N_constraints.txt'
# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# lg.generate_library(isolet_data, isolet_target, isolet_name, 160, isolet_class_num,
#                     n_cluster_lower_bound=isolet_class_num,
#                     n_cluster_upper_bound=10 * isolet_class_num,
#                     sampling_method='E2CP', constraints_file=isolet_constraints_file)

"""
ISOLET 160 Cop-KMeans ensemble (constraints=500)
"""
# isolet_name = 'ISOLET'
# isolet_constraints_file = 'Constraints/N_constraints.txt'
# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# lg.generate_library(isolet_data, isolet_target, isolet_name, 160, isolet_class_num,
#                     n_cluster_lower_bound=isolet_class_num,
#                     n_cluster_upper_bound=10 * isolet_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=isolet_constraints_file)

"""
ISOLET 160 E2CP ensemble (constraints=n)
"""
# isolet_name = 'ISOLET'
# isolet_constraints_file = 'Constraints/ISOLET_constraints_1520_2.txt'
# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# lg.generate_library(isolet_data, isolet_target, isolet_name, 160, isolet_class_num,
#                     n_cluster_lower_bound=isolet_class_num,
#                     n_cluster_upper_bound=10 * isolet_class_num,
#                     sampling_method='E2CP', constraints_file=isolet_constraints_file)

"""
ISOLET 160 Cop-KMeans ensemble (constraints=n)
"""
# isolet_name = 'ISOLET'
# isolet_constraints_file = 'Constraints/ISOLET_constraints_1520_2.txt'
# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# lg.generate_library(isolet_data, isolet_target, isolet_name, 160, isolet_class_num,
#                     n_cluster_lower_bound=isolet_class_num,
#                     n_cluster_upper_bound=10 * isolet_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=isolet_constraints_file)


"""
Generate size-1000 COIL20 library
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# coil_res_names = []
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(coil_data, coil_target, coil_name, 1000, coil_class_num,
#                             n_cluster_lower_bound=5 * coil_class_num,
#                             n_cluster_upper_bound=10 * coil_class_num,
#                             feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                             sampling_method='FSRSNC', generate_only=True)
#         coil_res_names.append(resname)
# el.evaluate_libraries_to_file(coil_res_names, 'Results/COIL20/', coil_class_num, coil_target
#                               ,'COIL.csv')
# el.do_eval_in_folder('_pure', 'Results/COIL20/', coil_class_num, coil_target, 'COIL.csv')

"""
Generate size-1000 MNIST4000 library
"""
# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# mnist_res_names = []
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(mnist_data, mnist_target, mnist_name, 1000, mnist_class_num,
#                             n_cluster_lower_bound=5 * mnist_class_num,
#                             n_cluster_upper_bound=10 * mnist_class_num,
#                             feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                             sampling_method='FSRSNC', generate_only=True)
#         mnist_res_names.append(resname)
# el.evaluate_libraries_to_file(mnist_res_names, 'Results/MNIST4000/', mnist_class_num, mnist_target
#                               ,'MNIST.csv')
# el.do_eval_in_folder('_pure', 'Results/MNIST4000/', mnist_class_num, mnist_target, 'MNIST.csv')


"""
Generate size-1000 isolet library
"""
# isolet_name = 'ISOLET'
# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# isolet_res_names = []
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(isolet_data, isolet_target, isolet_name, 1000, isolet_class_num,
#                                       n_cluster_lower_bound=5 * isolet_class_num,
#                                       n_cluster_upper_bound=10 * isolet_class_num,
#                                       feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                                       sampling_method='FSRSNC', generate_only=True)
#         isolet_res_names.append(resname)
# el.do_eval_in_folder('_pure', 'Results/ISOLET/', isolet_class_num, isolet_target, 'ISOLET.csv')
# print isolet_res_names


# isolet_data, isolet_target = dSP.loadIsolet()
# isolet_class_num = 26
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# el.do_eval_in_folder('_pure', 'Results/ISOLET/', isolet_class_num, isolet_target, 'ISOLET.csv')
# el.do_eval_in_folder('_pure', 'Results/COIL20/', coil_class_num, coil_target, 'COIL20.csv')
# el.do_eval_in_folder('_pure', 'Results/MNIST4000/', mnist_class_num, mnist_target, 'MNIST4000.csv')

"""
MNIST4000 1000 E2CP ensemble (constraints=1000)
"""
mnist_name = 'MNIST4000'
mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000_2.txt'
mnist_data, mnist_target = dSP.load_mnist_4000()
mnist_class_num = 10
lg.generate_library(mnist_data, mnist_target, mnist_name, 1000, mnist_class_num,
                    n_cluster_lower_bound=5 * mnist_class_num,
                    n_cluster_upper_bound=10 * mnist_class_num,
                    sampling_method='E2CP', constraints_file=mnist_constraints_file)

"""
MNIST4000 1000 Cop-KMeans ensemble (constraints=1000)
"""
# mnist_name = 'MNIST4000'
# mnist_constraints_file = 'Constraints/MNIST4000_constraints_1000.txt'
# mnist_data, mnist_target = dSP.load_mnist_4000()
# mnist_class_num = 10
# lg.generate_library(mnist_data, mnist_target, mnist_name, 1000, mnist_class_num,
#                     n_cluster_lower_bound=5 * mnist_class_num,
#                     n_cluster_upper_bound=10 * mnist_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=mnist_constraints_file)

"""
ISOLET 1000 E2CP ensemble (constraints=500)
"""
isolet_name = 'ISOLET'
isolet_constraints_file = 'Constraints/ISOLET_constraints_500_2.txt'
isolet_data, isolet_target = dSP.loadIsolet()
isolet_class_num = 26
lg.generate_library(isolet_data, isolet_target, isolet_name, 1000, isolet_class_num,
                    n_cluster_lower_bound=5 * isolet_class_num,
                    n_cluster_upper_bound=10 * isolet_class_num,
                    sampling_method='E2CP', constraints_file=isolet_constraints_file)


"""
ISOLET 1000 Cop-KMeans ensemble (constraints=500)
"""
isolet_name = 'ISOLET'
isolet_constraints_file = 'Constraints/ISOLET_constraints_500_2.txt'
isolet_data, isolet_target = dSP.loadIsolet()
isolet_class_num = 26
lg.generate_library(isolet_data, isolet_target, isolet_name, 1000, isolet_class_num,
                    n_cluster_lower_bound=5 * isolet_class_num,
                    n_cluster_upper_bound=10 * isolet_class_num,
                    sampling_method='Cop_KMeans', constraints_file=isolet_constraints_file)

"""
COIL20 1000 Cop-KMeans ensemble (constraints=500)
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# lg.generate_library(coil_data, coil_target, coil_name, 1000, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=coil_constraints_file)

"""
COIL20 1000 E2CP ensemble (constraints=500)
"""
# coil_name = 'COIL20'
# coil_constraints_file = 'Constraints/COIL20_constraints_500.txt'
# coil_data, coil_target = dSP.load_coil20()
# coil_class_num = 20
# lg.generate_library(coil_data, coil_target, coil_name, 1000, coil_class_num,
#                     n_cluster_lower_bound=5 * coil_class_num,
#                     n_cluster_upper_bound=10 * coil_class_num,
#                     sampling_method='E2CP', constraints_file=coil_constraints_file)

"""
Generate size-1000 OptDigits
"""
# optdigit_name = 'OPTDIGITS'
# optdigit_data, optdigit_target = dSP.load_digit()
# optdigit_class_num = 10
# # isolet_res_names = []
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(optdigit_data, optdigit_target, optdigit_name, 1000, optdigit_class_num,
#                                       n_cluster_lower_bound=5 * optdigit_class_num,
#                                       n_cluster_upper_bound=10 * optdigit_class_num,
#                                       feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                                       sampling_method='FSRSNC', generate_only=True)
#         # isolet_res_names.append(resname)
# el.do_eval_in_folder('_pure', 'Results/OPTDIGITS/', optdigit_class_num, optdigit_target, 'OPTDIGIT.csv')

"""
Generate size-1000 Segmentation
"""
# segmentation_name = 'segmentation'
# segmentation_data, segmentation_target = dSP.load_segmentation()
# segmentation_class_num = 7
# # isolet_res_names = []
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(segmentation_data, segmentation_target, segmentation_name, 1000, segmentation_class_num,
#                                       n_cluster_lower_bound=5 * segmentation_class_num,
#                                       n_cluster_upper_bound=10 * segmentation_class_num,
#                                       feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                                       sampling_method='FSRSNC', generate_only=True)
#         # isolet_res_names.append(resname)
# el.do_eval_in_folder('_pure', 'Results/segmentation/', segmentation_class_num, segmentation_target, 'segmentation.csv')

"""
Generate size-1000 WDBC
"""
# wdbc_name = 'WDBC'
# wdbc_data, wdbc_target = dSP.load_wdbc()
# wdbc_class_num = 2
# # isolet_res_names = []
# FSRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SSRs = [0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(0, 9):
#     for j in range(0, 5):
#         resname = lg.generate_library(wdbc_data, wdbc_target, wdbc_name, 1000, wdbc_class_num,
#                                       n_cluster_lower_bound=5 * wdbc_class_num,
#                                       n_cluster_upper_bound=10 * wdbc_class_num,
#                                       feature_sampling=FSRs[i], sample_sampling=SSRs[j],
#                                       sampling_method='FSRSNC', generate_only=True)
#         # isolet_res_names.append(resname)
# el.do_eval_in_folder('_pure', 'Results/WDBC/', wdbc_class_num, wdbc_target, 'WDBC.csv')

# """
# OPTDIGITS 1000 Cop-KMeans ensemble (constraints=1000)
# """
# optdigits_name = 'OPTDIGITS'
# optdigits_constraints_file = 'Constraints/OptDigits_constraints_1000_1.txt'
# optdigits_data, optdigits_target = dSP.load_digit()
# optdigits_class_num = 10
# lg.generate_library(optdigits_data, optdigits_target, optdigits_name, 1000, optdigits_class_num,
#                     n_cluster_lower_bound=5 * optdigits_class_num,
#                     n_cluster_upper_bound=10 * optdigits_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=optdigits_constraints_file)
#
# """
# OPTDIGITS 1000 E2CP ensemble (constraints=1000)
# """
# optdigits_name = 'OPTDIGITS'
# optdigits_constraints_file = 'Constraints/OptDigits_constraints_1000_1.txt'
# optdigits_data, optdigits_target = dSP.load_digit()
# optdigits_class_num = 10
# lg.generate_library(optdigits_data, optdigits_target, optdigits_name, 1000, optdigits_class_num,
#                     n_cluster_lower_bound=5 * optdigits_class_num,
#                     n_cluster_upper_bound=10 * optdigits_class_num,
#                     sampling_method='E2CP', constraints_file=optdigits_constraints_file)

"""
WDBC 1000 Cop-KMeans ensemble (constraints=200)
"""
# wdbc_name = 'WDBC'
# wdbc_constraints_file = 'Constraints/wdbc_constraints_200_1.txt'
# wdbc_data, wdbc_target = dSP.load_wdbc()
# wdbc_class_num = 2
# lg.generate_library(wdbc_data, wdbc_target, wdbc_name, 1000, wdbc_class_num,
#                     n_cluster_lower_bound=5 * wdbc_class_num,
#                     n_cluster_upper_bound=10 * wdbc_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=wdbc_constraints_file)

"""
WDBC 1000 E2CP ensemble (constraints=200)
"""
# wdbc_name = 'WDBC'
# wdbc_constraints_file = 'Constraints/wdbc_constraints_200_1.txt'
# wdbc_data, wdbc_target = dSP.load_wdbc()
# wdbc_class_num = 2
# lg.generate_library(wdbc_data, wdbc_target, wdbc_name, 1000, wdbc_class_num,
#                     n_cluster_lower_bound=5 * wdbc_class_num,
#                     n_cluster_upper_bound=10 * wdbc_class_num,
#                     sampling_method='E2CP', constraints_file=wdbc_constraints_file)

# """
# Segmentation 1000 Cop-KMeans ensemble (constraints=200)
# """
# segmentation_name = 'segmentation'
# segmentation_constraints_file = 'Constraints/segmentation_constraints_700_1.txt'
# segmentation_data, segmentation_target = dSP.load_segmentation()
# segmentation_class_num = 7
# lg.generate_library(segmentation_data, segmentation_target, segmentation_name, 1000, segmentation_class_num,
#                     n_cluster_lower_bound=5 * segmentation_class_num,
#                     n_cluster_upper_bound=10 * segmentation_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=segmentation_constraints_file)
#
# """
# Segmentation 1000 E2CP ensemble (constraints=200)
# """
# segmentation_name = 'segmentation'
# segmentation_constraints_file = 'Constraints/segmentation_constraints_700_1.txt'
# segmentation_data, segmentation_target = dSP.load_segmentation()
# segmentation_class_num = 7
# lg.generate_library(segmentation_data, segmentation_target, segmentation_name, 1000, segmentation_class_num,
#                     n_cluster_lower_bound=5 * segmentation_class_num,
#                     n_cluster_upper_bound=10 * segmentation_class_num,
#                     sampling_method='E2CP', constraints_file=segmentation_constraints_file)

"""
Segmentation 1000 Cop-KMeans ensemble (constraints=200)
"""
# segmentation_name = 'segmentation'
# segmentation_constraints_file = 'Constraints/segmentation_constraints_700_1.txt'
# segmentation_data, segmentation_target = dSP.load_segmentation()
# segmentation_class_num = 7
# lg.generate_library(segmentation_data, segmentation_target, segmentation_name, 1000, segmentation_class_num,
#                     n_cluster_lower_bound=5 * segmentation_class_num,
#                     n_cluster_upper_bound=10 * segmentation_class_num,
#                     sampling_method='Cop_KMeans', constraints_file=segmentation_constraints_file)

"""
Segmentation 1000 E2CP ensemble (constraints=200)
"""
# segmentation_name = 'segmentation'
# segmentation_constraints_file = 'Constraints/segmentation_constraints_700_1.txt'
# segmentation_data, segmentation_target = dSP.load_segmentation()
# segmentation_class_num = 7
# lg.generate_library(segmentation_data, segmentation_target, segmentation_name, 1000, segmentation_class_num,
#                     n_cluster_lower_bound=5 * segmentation_class_num,
#                     n_cluster_upper_bound=10 * segmentation_class_num,
#                     sampling_method='E2CP', constraints_file=segmentation_constraints_file)