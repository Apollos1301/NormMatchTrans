from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np


def locations_to_features_diffs(x_1, y_1, x_2, y_2):
    res = np.array([0.5 + 0.5 * (x_1 - x_2) / 384.0, 0.5 + 0.5 * (y_1 - y_2) / 384.0]) #change the normalization value according to your image size value
    return res


def build_graphs(P_np: np.ndarray, n: int, n_pad: int = None, edge_pad: int = None):

    A = delaunay_triangulate(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    edge_list = [[], []]
    
    # Vectorized edge and feature extraction to avoid slow O(n^2) nested loops
    row, col = np.nonzero(A[:n, :n])
    
    if len(row) > 0:
        edge_list = [row.tolist(), col.tolist()]
        x_1, y_1 = P_np[row, 0], P_np[row, 1]
        x_2, y_2 = P_np[col, 0], P_np[col, 1]
        features = np.stack([
            0.5 + 0.5 * (x_1 - x_2) / 384.0, 
            0.5 + 0.5 * (y_1 - y_2) / 384.0
        ], axis=1)
    else:
        features = np.zeros(shape=(0, 2))

    return np.array(edge_list, dtype=int), features


def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = np.ones((n, n)) - np.eye(n)
    else:
        try:
            d = Delaunay(P)
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            try:
                # Add QJ to joggle the input to avoid coplanar/collinear errors
                d = Delaunay(P, qhull_options="QJ")
                A = np.zeros((n, n))
                for simplex in d.simplices:
                    for pair in itertools.permutations(simplex, 2):
                        A[pair] = 1
            except QhullError as err2:
                # If it still fails, silently return a fully-connected graph (or optionally log it)
                A = np.ones((n, n)) - np.eye(n)
    return A
