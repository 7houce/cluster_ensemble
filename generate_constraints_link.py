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
