import random as rand
import numpy as np


def generateConstraints(target, n=0):
    """
    generate constraints for the given class labels

    Parameters
    ----------
    :param targets: given class labels
    :param n: number of constraints to generate

    Returns
    -------
    :return: must-link constraints and cannot-link constraints in 2 lists.
    """
    dataLength = len(target)
    must_link = []
    cannot_link = []
    cur_n = 0
    flag = True

    while flag:
        # choose sample randomly
        sample_1 = rand.randint(0, dataLength-1)
        sample_2 = rand.randint(0, dataLength-1)

        # we don't accept same sample to be the constraint
        # make the first sample to be the smaller one in order to filter the duplicates
        if sample_1 == sample_2:
            continue
        elif sample_1 > sample_2:
            temp = sample_1
            sample_1 = sample_2
            sample_2 = temp
        # filter the duplicates
        # if they are in the same class, append to the must-link set, or otherwise, the cannot-link set
        if (sample_1, sample_2) in must_link or (sample_1, sample_2) in cannot_link:
            continue
        elif target[sample_1] != target[sample_2]:
            cannot_link.append((sample_1, sample_2))
            cur_n += 1
        else:
            must_link.append((sample_1, sample_2))
            cur_n += 1
        if cur_n == n:
            flag = False

    return must_link, cannot_link


def generate_closure_constraints(targets, n=0):
    """
    generate transitive-closure constraints
    note that the number of constraints generated may be more than n

    Parameters
    ----------
    :param targets: real labels of given data set
    :param n: number of constraints to generate

    Returns
    -------
    :return: must-link constraints and cannot-link constraints in 2 lists.
    """
    data_len = len(targets)
    must_link = []
    cannot_link = []
    cur_n = 0
    stop = False
    ml_graph = dict()
    cl_graph = dict()

    for x in range(data_len):
        ml_graph[x] = set()
        cl_graph[x] = set()

    def add_both(d, i, j, ls):
        d[i].add(j)
        d[j].add(i)
        if i > j:
            tmp = i
            i = j
            j = tmp
        # make the first sample to be the smaller one in order to filter the duplicates
        ls.append((i, j))

    while not stop:
        # choose sample randomly
        samp1 = rand.randint(0, data_len-1)
        samp2 = rand.randint(0, data_len-1)

        # we don't accept same sample to be the constraint
        if samp1 == samp2:
            continue

        # filter the duplicates
        # if they are in the same class, append to the must-link set, or otherwise, the cannot-link set
        if (samp1, samp2) in must_link or (samp1, samp2) in cannot_link:
            continue
        elif targets[samp1] != targets[samp2]:
            add_both(cl_graph, samp1, samp2, cannot_link)
            cur_n += 1
            for x in ml_graph[samp1]:
                if x not in cl_graph[samp2]:
                    add_both(cl_graph, x, samp2, cannot_link)
                    cur_n += 1
            for y in ml_graph[samp2]:
                if y not in cl_graph[samp1]:
                    add_both(cl_graph, y, samp1, cannot_link)
                    cur_n += 1
        else:
            add_both(ml_graph, samp1, samp2, must_link)
            cur_n += 1
            for x in ml_graph[samp1]:
                if x not in ml_graph[samp2] and x != samp2:
                    add_both(ml_graph, x, samp2, must_link)
                    cur_n += 1
            for y in ml_graph[samp2]:
                if y not in ml_graph[samp1] and y != samp1:
                    add_both(ml_graph, y, samp1, must_link)
                    cur_n += 1
            for x in cl_graph[samp1]:
                if x not in cl_graph[samp2]:
                    add_both(cl_graph, x, samp2, cannot_link)
                    cur_n += 1
            for y in cl_graph[samp2]:
                if y not in cl_graph[samp1]:
                    add_both(cl_graph, y, samp1, cannot_link)
                    cur_n += 1

        if cur_n >= n:
            stop = True

    return must_link, cannot_link, ml_graph, cl_graph


def generate_closure_constraints_with_portion(targets, must_count=0, cannot_count=0):
    """
    generate transitive-closure constraints
    the number of must-link constraints generated in
    different classes is decided by their portion to all samples

    Parameters
    ----------
    :param targets: real labels of given data set
    :param must_count: number of must-link constraints to generate
    :param cannot_count: number of cannot-link constraints to generate

    Returns
    -------
    :return: must-link constraints and cannot-link constraints in 2 lists.
    """
    data_len = len(targets)
    clusters = np.unique(np.array(targets))
    n_must_link = [0] * len(clusters)
    must_link = []
    cannot_link = []
    ml_graph = dict()
    cl_graph = dict()

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
