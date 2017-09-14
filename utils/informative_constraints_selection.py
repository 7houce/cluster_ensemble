from __future__ import print_function
from sklearn.neighbors import NearestNeighbors
import utils.io_func as ioutil

_default_constraints_folder = 'Constraints/'


def _swap_in_order(first, second):
    """
    swap if first arg larger than second arg
    for de-duplication
    """
    if first >= second:
        return second, first
    else:
        return first, second


def get_cannot_link_from_nn(data, target, n_nbor=10, nn_algorithm='ball_tree'):
    """
    Get the 'informative' constraint set by finding
    those pairs of samples that close in distance metric
    but split into different clusters using k-nearest neighbours


    :param data: dataset
    :param target: labels
    :param n_nbor: number of nn, default to 10
    :param nn_algorithm: algorithm adopt used for constructing a nn-model, default to 'ball_tree'
    :return:
    """
    nbrs = NearestNeighbors(n_neighbors=n_nbor, algorithm=nn_algorithm).fit(data)
    distances, indices = nbrs.kneighbors(data)
    indices = indices[:, 1:]
    count = 0
    constraint_set = []
    for i, row in enumerate(indices):
        for element in row:
            if target[element] != target[i]:
                # de-duplication
                first, second = _swap_in_order(element, i)
                if (first, second) not in constraint_set:
                    constraint_set.append((first, second))
                    count += 1
    print('[get_cannot_link_from_nn]: '+str(count)+' informative cannot-link constraint(s) are found.')
    return constraint_set


def generate_informative_cl_set(data, target, dataset_name, n_nbor=10, nn_algorithm='ball_tree'):
    """
    generate the 'informative' constraint set and save to a file

    :param data: dataset
    :param target: labels
    :param dataset_name: name of dataset
    :param n_nbor: number of neighbours, default to 10
    :param nn_algorithm: algorithm adopt used for constructing a nn-model, default to 'ball_tree'
    :return:
    """
    constraints_set = get_cannot_link_from_nn(data, target, n_nbor=n_nbor, nn_algorithm=nn_algorithm)
    ioutil.store_constraints(_default_constraints_folder + dataset_name + '_informative_' + str(n_nbor),
                             [], constraints_set)
    return
