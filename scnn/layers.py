"""
Based on ChebyGCN (https://github.com/aclyde11/ChebyGCN),
`ChebyGCN/layers.py` file, commit a70d684c48c34e397b098290d7c6940953e0ccc2.
ChebyGCN is based on cnn_graph (https://github.com/mdeff/cnn_graph).

Copyright (c) 2016 Michaël Defferrard
Copyright (c) 2019 Austin Clyde
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

# Based on Michael Defferrard, Xavier Bresson, Pierre Vandergheynst,
# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering,
# Neural Information Processing Systems (NIPS), 2016.

import tensorflow as tf

import numpy as np
import scipy
import healpy as hp

from . import utils


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, filter_size, poly_k, L=None, bias_per_vertex=False, **kwargs):
        self.F_1 = filter_size
        self.K = poly_k
        self.L = L
        self.output_dim = ()
        self.bias_per_vertex = bias_per_vertex
        self.M_0 = None
        self.kernel = None
        self.bias = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.M_0 = input_shape[1]
        F_0 = input_shape[2]
        # Xavier-like weight initializer for Chebyshev coefficients
        stddev = 1 / np.sqrt(int(F_0) * (self.K + 0.5) / 2)
        initializer = tf.initializers.TruncatedNormal(0, stddev=stddev)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(F_0 * self.K, self.F_1),
            initializer=initializer,
            trainable=True,
        )
        if self.bias_per_vertex:
            self.bias = self.add_weight(
                name="bias",
                shape=(1, self.M_0, self.F_1),
                initializer="uniform",
                trainable=True,
            )
        else:
            self.bias = self.add_weight(
                name="bias",
                shape=(1, 1, self.F_1),
                initializer="uniform",
                trainable=True,
            )
        super().build(input_shape)  # Be sure to call this at the end

    @staticmethod
    def rescale_L(L, lmax=2, scale=1):
        """Rescale the Laplacian eigenvalues in [-scale,scale]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format="csr", dtype=L.dtype)
        L *= 2 * scale / lmax
        return L - I

    def chebyshev5(self, x, L):
        shape = tf.shape(input=x)
        M = shape[1]
        Fin = shape[2]
        # Transform to Chebyshev basis
        x0 = tf.transpose(a=x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K_coeff x M x Fin*N

        if self.K > 1:
            x1 = tf.sparse.sparse_dense_matmul(L, x0)
            x = concat(x, x1)
        for _ in range(2, self.K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K_coeff x M x Fin x N
        x = tf.transpose(a=x, perm=[3, 1, 2, 0])  # N x M x Fin x K_coeff
        x = tf.reshape(x, [-1, Fin * self.K])  # N*M x Fin*K_coeff
        # Filter: Fin*Fout filters of order K_coeff, i.e., one filterbank per feature pair
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        return tf.reshape(x, [-1, M, self.F_1])  # N x M x Fout

    def call(self, x):
        if len(x.get_shape()) != 3:
            x = tf.expand_dims(x, 2)
        x = self.chebyshev5(x, self.L)
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M_0, self.F_1)

    def get_config(self):
        config = {
            "filter_size": self.F_1,
            "poly_k": self.K,
            "bias_per_vertex": self.bias_per_vertex,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SphereConvolution(GraphConvolution):
    def __init__(self, filter_size, poly_k, nside=None, **kwargs):
        """
        Initialize spherical convolution layer.

        Parameters
        ----------
        filter_size :
            Size of Laplacian filter
        poly_k :
            Order of Chebyshev polynomial
        nside :
            HEALPix nside parameter.
            Optional if input covers complete sphere, else required.
        """
        self.nside = nside
        super().__init__(filter_size, poly_k, **kwargs)

    def build(self, input_shape):
        nside = self.nside if self.nside else hp.npix2nside(int(input_shape[1]))

        # Rescale Laplacian and store as a TF sparse tensor
        L = utils.healpix_laplacian(nside)
        L = scipy.sparse.csr_matrix(L)
        lmax = (
            1.02
            * scipy.sparse.linalg.eigsh(L, k=1, which="LM", return_eigenvectors=False)[
                0
            ]
        )
        L = self.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        order = np.lexsort(
            (L.col, L.row)
        )  # Ensure row-major order (check probably not needed)
        self.L = tf.SparseTensor(indices[order], L.data[order], L.shape)

        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = {
            "nside": self.nside,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphPool(tf.keras.layers.Layer):
    def __init__(self, pooling, pool_type="max", **kwargs):

        self.p_1 = pooling
        self.pool_type = pool_type
        if pool_type == "max":
            self.poolf = tf.nn.max_pool2d
        elif pool_type == "average" or pool_type == "avg":
            self.poolf = tf.nn.avg_pool2d
        else:
            raise ValueError('pool_type not set to "max" or "avg"')

        super().__init__(**kwargs)

    def pool(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = self.poolf(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding="SAME")
            return tf.squeeze(x, [3])  # N x M/p x F
        return x

    def call(self, x):
        return self.pool(x, self.p_1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // self.p_1, input_shape[2])

    def get_config(self):
        config = {
            "pooling": self.p_1,
            "pool_type": self.pool_type,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LinearCombination(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=(), initializer="ones", dtype="float32", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(), initializer="zeros", dtype="float32", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return tf.add(self.bias, tf.multiply(self.kernel, x))

    def compute_output_shape(self, input_shape):
        return input_shape
