import numpy as np

_delimiter = ','


def read_matrix(filename):
    return np.loadtxt(filename, delimiter=_delimiter)


def store_matrix(mat, filename, fmt):
    np.savetxt(filename, mat, fmt=fmt, delimiter=_delimiter)


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
    return np.array(ml), np.array(cl)


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
