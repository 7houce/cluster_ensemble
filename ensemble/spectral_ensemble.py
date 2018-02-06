from __future__ import print_function
from sklearn import cluster
from ensemble.Cluster_Ensembles import build_hypergraph_adjacency
from ensemble.Cluster_Ensembles import build_weighted_hypergraph_adjacency
from ensemble.Cluster_Ensembles import build_weighted_hypergraph_adjacency_extended
import scipy.sparse
import numpy as np
import time as tt

_assign_labels = 'discretize'


def spectral_ensemble(base_clusterings, N_clusters_max=3, n_spec_init=1):
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]

    print("INFO: Spectral Ensemble Starting...")
    print("***************************************")
    print("Ensemble member :" + str(n_sols) + " solutions")
    print("Sample :" + str(n_samples))

    # build hyper-graph matrix
    adjc = build_hypergraph_adjacency(base_clusterings)
    adjc = adjc.transpose()

    att_mat = adjc.dot(adjc.transpose())
    att_mat = np.squeeze(np.asarray(att_mat.todense()))

    spec_ensembler = cluster.SpectralClustering(n_clusters=N_clusters_max, n_init=n_spec_init, affinity='precomputed')

    # spec_ensembler = cluster.SpectralClustering(n_clusters=N_clusters_max, n_init=n_spec_init, affinity='precomputed',
    #                                             assign_labels=_assign_labels)

    spec_ensembler.fit(att_mat)

    return spec_ensembler.labels_


def spectral_ensemble_on_matrix(coas_matrix, N_runs, N_clusters_max, n_spec_init=10):
    att_mat = coas_matrix
    spec_ensembler = cluster.SpectralClustering(n_clusters=N_clusters_max, n_init=n_spec_init, affinity='precomputed',
                                                assign_labels=_assign_labels)
    spec_ensembler.fit(att_mat)
    return spec_ensembler.labels_


def weighted_spectral_ensemble(base_clusterings, N_clusters_max=3, n_spec_init=10, weighted=True, clustering_weights=None,
                               cluster_level_weights=None, alpha=None, new_formula=False, internal=None, ml=None, cl=None):
    # get basic information
    n_sols = base_clusterings.shape[0]
    n_samples = base_clusterings.shape[1]

    print("INFO: Weighted Spectral Ensemble Starting...")
    print("***************************************")
    print("Ensemble member :" + str(n_sols) + " solutions")
    print("Sample :" + str(n_samples))

    if not weighted:
        raise Exception("Weighted Spectral Ensemble!")

    # build hyper-graph matrix
    # adjc = build_hypergraph_adjacency(base_clusterings)
    if new_formula and alpha is not None:
        print("Extended Weighted Ensemble: Spectral")
        adjc = build_weighted_hypergraph_adjacency_extended(base_clusterings,
                                                            clustering_weights,
                                                            cluster_level_weights,
                                                            alpha,
                                                            internal)
    else:
        adjc = build_weighted_hypergraph_adjacency(base_clusterings, clustering_weights, cluster_level_weights, alpha)

    att_mat = scipy.sparse.csr_matrix.dot(adjc.transpose().tocsr(),
                                    scipy.sparse.csr_matrix(([1] * adjc.data.size,
                                                             adjc.indices,
                                                             adjc.indptr),
                                                            shape=adjc.shape))

    att_mat = np.squeeze(np.asarray(att_mat.todense()))

    if ml is not None and cl is not None:
        min_vv = np.min(att_mat)
        max_vv = np.max(att_mat)
        for obj1, obj2 in ml:
            att_mat[obj1, obj2] = max_vv
            att_mat[obj2, obj1] = max_vv
        for obj1, obj2 in cl:
            att_mat[obj1, obj2] = min_vv
            att_mat[obj2, obj1] = min_vv

    spec_ensembler = cluster.SpectralClustering(n_clusters=N_clusters_max, n_init=n_spec_init, affinity='precomputed',
                                                assign_labels=_assign_labels)

    spec_ensembler.fit(att_mat)

    return spec_ensembler.labels_
