"""
Utilities module.

Based on DeepSphere (https://github.com/SwissDataScienceCenter/DeepSphere),
`deepsphere/utils.py` file, commit 262573f12c8a7eac058ac85f520401da77b380af.

Copyright (c) 2018 Nathanaël Perraudin, Michaël Defferrard
Copyright (c) 2019-2020 Matthew Petroff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import scipy.sparse
import healpy as hp


# Copied from https://github.com/deepsphere/paper-iclr20/blob/51260d9169b9bff2f0d71d567c99909a17efd5e9/figures/kernel_widths.py
KERNEL_WIDTH_OPTIMAL = {
    32: 0.02500,
    64: 0.01228,
    128: 0.00614,
    256: 0.00307,
    512: 0.00154,
    1024: 0.00077,
}
KERNEL_WIDTH_FIT = np.poly1d(
    np.polyfit(
        np.log(sorted(KERNEL_WIDTH_OPTIMAL.keys())),
        np.log(
            [i[1] for i in sorted(KERNEL_WIDTH_OPTIMAL.items(), key=lambda i: i[0])]
        ),
        1,
    )
)
for nside in [1, 2, 4, 8, 16]:
    # Populate kernel widths for other map sizes
    KERNEL_WIDTH_OPTIMAL[nside] = np.exp(KERNEL_WIDTH_FIT(np.log(nside)))


def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    indexes : list of int, optional
        List of indexes to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    """
    if not nest:
        raise NotImplementedError()

    if indexes is None:
        indexes = range(nside ** 2 * 12)
    npix = len(indexes)  # Number of pixels.
    if npix >= (max(indexes) + 1):
        # If the user input is not consecutive nodes, we need to use a slower method
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    # Get the coordinates
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
    # Indices of non-zero values in the adjacency matrix
    col_index = neighbors.T.reshape((npix * 8))
    row_index = np.repeat(indexes, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere)
    if usefast:
        keep = col_index < npix
        # Remove fake neighbors (some pixels have less than 8)
        keep &= col_index >= 0
        col_index = col_index[keep]
        row_index = row_index[keep]
    else:
        col_index_set = set(indexes)
        keep = [c in col_index_set for c in col_index]
        inverse_map = [np.nan] * (nside ** 2 * 12)
        for i, index in enumerate(indexes):
            inverse_map[index] = i
        col_index = [inverse_map[el] for el in col_index[keep]]
        row_index = [inverse_map[el] for el in row_index[keep]]

    # Compute Euclidean distances between neighbors
    distances = np.sum((coords[row_index] - coords[col_index]) ** 2, axis=1)

    # Compute similarities / edge weights
    # kernel_width = np.mean(distances) / 2  # Perraudin et al. 2019
    kernel_width = KERNEL_WIDTH_OPTIMAL[nside] ** 2 / (
        4 * np.log(2)
    )  # Defferrard et al. 2020
    weights = np.exp(-distances / (4 * kernel_width))

    # Build the sparse matrix
    W = scipy.sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype
    )

    return W


def build_laplacian(W, lap_type="normalized", dtype=np.float32):
    """Build a Laplacian (TensorFlow)."""
    d = np.ravel(W.sum(1))
    if lap_type == "combinatorial":
        D = scipy.sparse.diags(d, 0, dtype=dtype)
        return (D - W).tocsc()
    elif lap_type == "normalized":
        d12 = np.power(d, -0.5)
        D12 = scipy.sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return scipy.sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
    else:
        raise ValueError("Unknown Laplacian type {}".format(lap_type))


def healpix_laplacian(
    nside=16, nest=True, lap_type="normalized", indexes=None, dtype=np.float32
):
    """Build a Healpix Laplacian."""
    W = healpix_weightmatrix(nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    L = build_laplacian(W, lap_type=lap_type)
    return L
