import random as rand


def generateConstraints(targets, n=0):
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
    data_len = len(targets)
    must_link = []
    cannot_link = []
    cur_n = 0
    stop = False

    while not stop:
        # choose sample randomly
        samp1 = rand.randint(0, data_len-1)
        samp2 = rand.randint(0, data_len-1)

        # we don't accept same sample to be the constraint
        # make the first sample to be the smaller one in order to filter the duplicates
        if samp1 == samp2:
            continue
        elif samp1 > samp2:
            temp = samp1
            samp1 = samp2
            samp2 = temp

        # filter the duplicates
        # if they are in the same class, append to the must-link set, or otherwise, the cannot-link set
        if (samp1, samp2) in must_link or (samp1, samp2) in cannot_link:
            continue
        elif targets[samp1] != targets[samp2]:
            cannot_link.append((samp1, samp2))
            cur_n += 1
        else:
            must_link.append((samp1, samp2))
            cur_n += 1
        if cur_n == n:
            stop = True

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


def read_constraints(consfile):
    """
    read constraints from file to two list

    Parameters
    ----------
    :param consfile: file that stores the constraints

    Returns
    -------
    :return: must-link set and cannot-link set as list
    """
    ml, cl = [], []
    with open(consfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                line = line.split()
                constraint = (int(line[0]), int(line[1]))
                c = int(line[2])
                if c == 1:
                    ml.append(constraint)
                if c == -1:
                    cl.append(constraint)
    return ml, cl


def store_constraints(consfile, mlset, nlset):
    """
    store generated constraints into given file

    Parameters
    ----------
    :param consfile: file to store the constraints
    :param mlset: must-link set
    :param nlset: cannot-link set
    """
    with open(consfile, 'w') as f:
        str_list = []
        for i, j in mlset:
            str_list.append(str(i) + ' ' + str(j) + ' 1' + '\n')
        for i, j in nlset:
            str_list.append(str(i) + ' ' + str(j) + ' -1' + '\n')
        f.writelines(str_list)
    return
