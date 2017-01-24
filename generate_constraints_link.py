import random as rand


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
