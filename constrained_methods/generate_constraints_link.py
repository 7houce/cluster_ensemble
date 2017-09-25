import random as rand
import numpy as np
import os
import utils.io_func as io_func
import utils.exp_datasets as exd
import utils.informative_constraints_selection as ics

_default_constraints_postfix = ['constraints_quarter_n', 'constraints_half_n', 'constraints_n', 'constraints_onehalf_n',
                                'constraints_2n']
_default_constraints_folder = 'Constraints/'
_default_diff_portion = 1
_default_diff_portion_name = 'diff_n'
_default_informative_postfix = '_informative'


def _build_default_amount_array(n_samples):
    array = range(int(0.5 * n_samples), 2 * n_samples + 1, int(0.5 * n_samples))
    array.insert(0, int(0.25 * n_samples))
    return array


def generate_closure_constraints_with_portion(dataset_name, must_count=0, cannot_count=0,
                                              informative=False):
    """
    generate transitive-closure constraints
    the number of must-link constraints generated in
    different classes is decided by their portion to all samples

    Parameters
    ----------
    :param targets: real labels of given data set
    :param dataset_name:
    :param must_count: number of must-link constraints to generate
    :param cannot_count: number of cannot-link constraints to generate
    :param informative:

    Returns
    -------
    :return: must-link constraints and cannot-link constraints in 2 lists.
    """
    data, targets = exd.dataset[dataset_name]['data']()
    data_len = len(targets)
    clusters = np.unique(np.array(targets))
    n_must_link = [0] * len(clusters)
    must_link = []
    cannot_link = []
    ml_graph = dict()
    cl_graph = dict()

    if informative:
        if os.path.isfile(_default_constraints_folder + dataset_name + '_informative_constraints.txt'):
            print 'informative constraints already existed.'
            _, informative_cl = io_func.read_constraints(
                _default_constraints_folder + dataset_name + '_informative_constraints.txt')
        else:
            print 'informative constraints not exist, generating...'
            ics.generate_informative_cl_set(dataset_name)
            _, informative_cl = io_func.read_constraints(
                _default_constraints_folder + dataset_name + '_informative_constraints.txt')
        informative_len = len(informative_cl)

    for x in range(data_len):
        ml_graph[x] = set()
        cl_graph[x] = set()

    for cluster in clusters:
        n_must_link[cluster] = int(len(targets[targets == cluster]) / float(data_len) * must_count)

    print n_must_link

    def add_both(d, i, j, ls):
        d[i].add(j)
        d[j].add(i)
        if i > j:
            tmp = i
            i = j
            j = tmp
        # make the first sample to be the smaller one in order to filter the duplicates
        ls.append((i, j))

    for cluster in clusters:
        all_samples = np.where(targets == cluster)
        all_samples = np.squeeze(all_samples)
        print all_samples
        cur_count = 0
        while cur_count < n_must_link[cluster]:
            selected_sample = np.random.choice(all_samples, 2)
            if selected_sample[0] > selected_sample[1]:
                temp = selected_sample[0]
                selected_sample[0] = selected_sample[1]
                selected_sample[1] = temp
            if (selected_sample[0], selected_sample[1]) in must_link:
                continue
            else:
                add_both(ml_graph, selected_sample[0], selected_sample[1], must_link)
                cur_count += 1
                for x in ml_graph[selected_sample[0]]:
                    if x not in ml_graph[selected_sample[1]] and x != selected_sample[1]:
                        add_both(ml_graph, x, selected_sample[1], must_link)
                        cur_count += 1
                for y in ml_graph[selected_sample[1]]:
                    if y not in ml_graph[selected_sample[0]] and y != selected_sample[0]:
                        add_both(ml_graph, y, selected_sample[0], must_link)
                        cur_count += 1

    cur_count = 0
    while cur_count < cannot_count:
        # choose sample randomly
        if informative:
            cl_tuple = informative_cl[rand.randint(0, informative_len-1)]
            samp1 = cl_tuple[0]
            samp2 = cl_tuple[1]
        else:
            samp1 = rand.randint(0, data_len-1)
            samp2 = rand.randint(0, data_len-1)

        # we don't accept same sample to be the constraint
        if samp1 == samp2 or targets[samp1] == targets[samp2]:
            continue

        if samp1 > samp2:
            temp = samp1
            samp1 = samp2
            samp2 = temp

        # filter the duplicates
        # if they are in the same class, append to the must-link set, or otherwise, the cannot-link set
        if (samp1, samp2) in must_link or (samp1, samp2) in cannot_link:
            continue
        else:
            add_both(cl_graph, samp1, samp2, cannot_link)
            cur_count += 1
            for x in ml_graph[samp1]:
                if x not in cl_graph[samp2]:
                    add_both(cl_graph, x, samp2, cannot_link)
                    cur_count += 1
            for y in ml_graph[samp2]:
                if y not in cl_graph[samp1]:
                    add_both(cl_graph, y, samp1, cannot_link)
                    cur_count += 1

    return must_link, cannot_link, ml_graph, cl_graph


def generate_diff_amount_constraints_wrapper(dataset_name, amount_intervals=None, postfixes=None, informative=False):
    """
    generate constraints of given dataset in different amount
    intervals of amount default to [0.25n, 0.5n, n, 1.5n, 2n],
    and default postfixes are also predefined in a human readable format.
    Notice that the length of amount_intervals and postfixes should be the same.

    :param dataset_name: name of the dataset, should be defined in exp_datasets
    :param amount_intervals: intervals of the amount of contraints as a list, default to [0.25n, 0.5n, n, 1.5n, 2n]
    :param postfixes: postfixes of the file storing the constraints, should be in the same length with intervals
    :param informative: informative cannot-link are adopted or not, default to False
    :return:
    """
    data, targets = exd.dataset[dataset_name]['data']()
    amount_array = amount_intervals if amount_intervals is not None else _build_default_amount_array(len(targets))
    postfix_array = postfixes if postfixes is not None else _default_constraints_postfix
    additional_postfix = _default_informative_postfix if informative else ''
    if len(amount_array) != len(postfix_array):
        raise ValueError('Length of amount and postfix should be the same.')
    for amount_value, postfix_value in zip(amount_array, postfix_array):
        ml, cl, _1, _2 = generate_closure_constraints_with_portion(dataset_name,
                                                                   must_count=int(amount_value/2),
                                                                   cannot_count=int(amount_value/2),
                                                                   informative=informative)
        io_func.store_constraints(
            _default_constraints_folder + dataset_name + '_' + postfix_value + additional_postfix + '.txt', ml, cl)
    return


def generate_diff_constraints_wrapper(dataset_name, amount=0, n_groups=0, postfix=None, informative=False):
    """
    generate different set of constraints in same amount

    :param dataset_name: name of the dataset, should be defined in exp_datasets
    :param amount: amount of constraints in one set, default to n
    :param n_groups: number of constraints set, default to 5
    :param postfix: postfix of constraints file, default to 'diff_n'
    :param informative: informative cannot-link are adopted or not, default to False
    :return:
    """
    data, targets = exd.dataset[dataset_name]['data']()
    if amount != 0 and postfix is None:
        raise ValueError('Postfix should be given while the amount of constraints is specified.')
    amount_value = amount if amount != 0 else int(len(targets) * _default_diff_portion)
    postfix_value = postfix if postfix is not None else _default_diff_portion_name
    groups = n_groups if n_groups != 0 else 5
    additional_postfix = _default_informative_postfix if informative else ''
    for i in range(1, groups + 1):
        ml, cl, _1, _2 = generate_closure_constraints_with_portion(dataset_name,
                                                                   must_count=int(amount_value / 2),
                                                                   cannot_count=int(amount_value / 2),
                                                                   informative=informative)
        io_func.store_constraints(
            _default_constraints_folder + dataset_name + '_' + postfix_value + '_' + str(
                i) + additional_postfix + '.txt', ml, cl)
    return


# def add_noise_to_constraints(ml, cl, target, portion=0):
#     n_samples = len(target)
#     if portion >= 1:
#         must_noise_num = int(portion / 2)
#         cannot_noise_num = int(portion / 2)
#     else:
#         must_noise_num = int(len(ml) * portion)
#         cannot_noise_num = int(len(cl) * portion)
#     while not stop:
#         # choose sample randomly
#         samp1 = rand.randint(0, n_samples-1)
#         samp2 = rand.randint(0, n_samples-1)
#         if target[]