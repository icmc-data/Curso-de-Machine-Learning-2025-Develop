"""
Sample code automatically generated on 2025-05-07 17:58:45

by www.matrixcalculus.org

from input

d/dF (1/2)*((norm2(R- 5*((exp((-1)*(F*U'))+matrix(1)).^(-1)) ))^2 + a*(norm2(F))^2) = a*F-10/2*((R-5*(matrix(1)+exp(-F*U')).^(-1)).*(matrix(1)+exp(-F*U')).^(-2).*exp(-F*U'))*U

where

F is a matrix
R is a matrix
U is a matrix
a is a scalar

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def grandient_F(F, R, U, a):
    assert isinstance(F, np.ndarray)
    dim = F.shape
    assert len(dim) == 2
    F_rows = dim[0]
    F_cols = dim[1]
    assert isinstance(R, np.ndarray)
    dim = R.shape
    assert len(dim) == 2
    R_rows = dim[0]
    R_cols = dim[1]
    assert isinstance(U, np.ndarray)
    dim = U.shape
    assert len(dim) == 2
    U_rows = dim[0]
    U_cols = dim[1]
    if isinstance(a, np.ndarray):
        dim = a.shape
        assert dim == (1, )
    assert F_rows == R_rows
    assert U_rows == R_cols
    assert F_cols == U_cols

    T_0 = np.exp(-(F).dot(U.T))
    T_1 = (np.ones((F_rows, U_rows)) + T_0)
    T_2 = (R - (5 * (T_1 ** -1)))
    gradient = ((a * F) - ((10 / 2) * (((T_2 * (T_1 ** -2)) * T_0)).dot(U)))

    return gradient

def checkGradient(F, R, U, a):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(F + t * delta, R, U, a)
    f2, _ = fAndG(F - t * delta, R, U, a)
    f, g = fAndG(F, R, U, a)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    F = np.random.randn(3, 3)
    R = np.random.randn(3, 3)
    U = np.random.randn(3, 3)
    a = np.random.randn(1)

    return F, R, U, a

if __name__ == '__main__':
    F, R, U, a = generateRandomData()
    functionValue, gradient = fAndG(F, R, U, a)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(F, R, U, a)
