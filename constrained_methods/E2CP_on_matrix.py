import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering

_k_E2CP = 15
_alpha = 0.1


def fit_constrained(similarity_matrix, ML, CL, n_clusters):
    N = similarity_matrix.shape[0]

    nbrs = NearestNeighbors(n_neighbors=_k_E2CP + 1,
                            algorithm='brute').fit(similarity_matrix)
    distances, indices = nbrs.kneighbors()
    W = np.zeros(similarity_matrix.shape)

    ind1 = (np.arange(N).reshape((-1, 1)) * np.ones((1, _k_E2CP))).reshape(-1).astype('int')
    ind2 = indices[:, 1:].reshape(-1).astype('int')
    W[ind1, ind2] = similarity_matrix[ind1, ind2] / (np.sqrt(similarity_matrix[ind1, ind1]) * np.sqrt(similarity_matrix[ind2, ind2]))

    W = (W + W.transpose()) / 2
    Dsqrt = np.diag(np.sum(W, axis=1) ** -0.5)
    Lbar = np.dot(np.dot(Dsqrt, W), Dsqrt)

    Z = np.zeros(similarity_matrix)
    Z[ML[:, 0], ML[:, 1]] = 1
    Z[CL[:, 0], CL[:, 1]] = -1

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
    temp = (1 - _alpha) * (np.eye(Lbar.shape[0]) - _alpha * Lbar)
    Fbar = np.dot(np.dot(temp, Z), temp.conj().T)

    Fbar = Fbar / np.max(np.abs(Fbar.reshape(-1)))

    Wbar = np.zeros(similarity_matrix)
    mlInd = Fbar >= 0
    Wbar[mlInd] = 1 - (1 - Fbar[mlInd]) * (1 - W[mlInd])
    clInd = Fbar < 0
    Wbar[clInd] = (1 + Fbar[clInd]) * W[clInd]

    specClus = SpectralClustering(n_clusters=n_clusters,
                                  affinity='precomputed')
    specClus.fit(Wbar)
    return specClus.labels_
