import numpy as np
import ensemble.Cluster_Ensembles as ce
import ensemble.spectral_ensemble as spec
from utils import io_func
import evaluation.Metrics as Metrics

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPA_on_matrix,
                    'Spectral': spec.spectral_ensemble_on_matrix}
_default_ensemble_method = ['CSPA', 'Spectral']


def propagation_on_coassociation_matrix(similarity_matrix, ML, CL, alpha):
    """
    apply constraint-propagation clustering e2cp on a given matrix.

    :param similarity_matrix: similarity matrix or affinity matrix of the dataset
    :param ML: must-link constraint set at the format of [[xx, yy], [yy, zz] .... ]
    :param CL: cannot-link constraint set
    :param n_clusters: #clusters
    :return: propagated co-association matrix
    """
    N = similarity_matrix.shape[0]
    similarity_matrix /= np.max(similarity_matrix)
    W = similarity_matrix

    print similarity_matrix.shape

    # expand the constraints set
    ML = np.append(ML, ML[:, -1::-1], axis=0)
    CL = np.append(CL, CL[:, -1::-1], axis=0)

    W = (W + W.transpose()) / 2
    Dsqrt = np.diag(np.sum(W, axis=1) ** -0.5)
    Lbar = np.dot(np.dot(Dsqrt, W), Dsqrt)

    Z = np.zeros(similarity_matrix.shape)
    Z[ML[:, 0], ML[:, 1]] = 1
    Z[CL[:, 0], CL[:, 1]] = -1

    # iterative approach
    # Fv = np.zeros(Z.shape)
    # for i in range(50):
    #     Fv = self.alpha * np.dot(Lbar, Fv) + (1 - self.alpha) * Z
    #
    # Fh = np.zeros(Z.shape)
    # for i in range(50):
    #     Fh = self.alpha * np.dot(Fh, Lbar) + (1 - self.alpha) * Fv
    #
    # Fbar = Fh / np.max(np.abs(Fh.reshape(-1)))

    # approximation of Fbar instead of the propagation iteration.
    temp = (1 - alpha) * np.linalg.inv((np.eye(Lbar.shape[0]) - alpha * Lbar))
    Fbar = np.dot(np.dot(temp, Z), temp.conj().T)

    Fbar = Fbar / np.max(np.abs(Fbar.reshape(-1)))

    # recover
    Wbar = np.zeros(similarity_matrix.shape)
    mlInd = Fbar >= 0
    Wbar[mlInd] = 1 - (1 - Fbar[mlInd]) * (1 - W[mlInd])
    clInd = Fbar < 0
    Wbar[clInd] = (1 + Fbar[clInd]) * W[clInd]

    # specClus = SpectralClustering(n_clusters=n_clusters,
    #                               affinity='precomputed')
    # specClus.fit(Wbar)
    return Wbar


def do_propagation_ensemble(library_folder, library_name, class_num, target,
                            constraint_file, logger, alphas, have_zero=True,
                            ensemble_method=_default_ensemble_method):
    logger.debug('===========================================================================================')
    logger.debug('-----------------Propagation Ensemble for library:' + str(library_name) + '----------------')
    logger.debug('-----------------Have zero type = ' + str(have_zero) + '-----------------------------------')
    logger.debug('-----------------Constraint File name = ' + constraint_file + '----------------------------')

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    ml, cl = io_func.read_constraints(constraint_file)

    hyperedges = ce.build_hypergraph_adjacency(labels)
    hyperedges = hyperedges.transpose()

    coas_matrix = hyperedges.dot(hyperedges.transpose())
    coas_matrix = np.squeeze(np.asarray(coas_matrix.todense()))
    coas_matrix = coas_matrix.astype(np.float32)
    coas_matrix /= np.max(coas_matrix)

    print coas_matrix

    nmis = []
    for alpha in alphas:
        logger.debug('-------------------------->>>>>> PARAM START <<<<<<<---------------------------------')
        propagated_coas_matrix = propagation_on_coassociation_matrix(coas_matrix, ml, cl, alpha)
        cur_nmis = []
        for method in ensemble_method:
            ensemble_label = _ensemble_method[method](propagated_coas_matrix, labels.shape[0], class_num)
            ensemble_nmi = Metrics.normalized_max_mutual_info_score(ensemble_label, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        logger.debug('------------------------->>>>>> END OF THIS PARAM <<<<<<------------------------------')
    logger.debug('===========================================================================================')
    return nmis
