import time
import torch
from torch import sparse

class HyperG:
    def __init__(self, H, X=None, w=None):
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]


        if X is not None:
            assert isinstance(X, torch.Tensor) and X.ndim == 2
            self._X = X
        else:
            self._X = None

        if w is not None:
            self.w = w.view(-1)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = torch.ones(self._n_edges)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X

    def node_degrees(self):
        if self._DV is None:
            H = self._H.tocsr()
            dv = H.matmul(self.w.view(-1, 1)).view(-1)
            self._DV = sparse.diags(dv, shape=(self._n_nodes, self._n_nodes))
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            H = self._H.tocsr()
            de = H.sum(dim=0).A.view(-1)
            self._DE = sparse.diags(de, shape=(self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            self.edge_degrees()
            inv_de = torch.pow(self._DE.data.view(-1), -1.)
            self._INVDE = sparse.diags(inv_de, shape=(self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            self.node_degrees()
            dv2 = torch.pow(self._DV.data.view(-1), -0.5)
            self._DV2 = sparse.diags(dv2, shape=(self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            self.inv_square_node_degrees()
            self.inv_edge_degrees()

            W = sparse.diags(self.w)
            self._THETA = self._DV2.matmul(self._H).matmul(W).matmul(self._INVDE).matmul(self._H.T).matmul(self._DV2)

        return self._THETA

    def laplacian(self):
        if self._L is None:
            self.theta_matrix()
            self._L = sparse.eye(self._n_nodes) - self._THETA
        return self._L

    def update_hyperedge_weights(self, w):
        assert isinstance(w, (torch.Tensor, list)), "The hyperedge array should be a torch.Tensor or list"

        self.w = torch.tensor(w).view(-1)
        assert w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def update_incident_matrix(self, H):
        assert sparse.issparse(H)
        assert H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges

        # TODO: reset hyperedge weights?

        self._H = H
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None


class IMHL:
    def __init__(self, M, omega):
        assert isinstance(M, list)
        assert isinstance(omega, torch.Tensor)
        assert len(M) == omega.shape[0]

        self.M = M
        self.omega = omega
