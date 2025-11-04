"""
Bernstein 5th order polynomial fitting and prediction package. 

Author: Emily Guan
"""

import numpy as np
import numpy.linalg as LA
import math

"""Return the binomial coefficient C(5, k)."""
def coefficients(k):
    return math.comb(5, k)

"""Evaluate the degree-5 Bernstein basis function B_{5,k}(t)."""
def basis(k, t):
    t = np.asarray(t, dtype=float)
    return coefficients(k) * (t ** k) * ((1 - t) ** (5 - k))

"""Stack all 6 degree-5 Bernstein basis functions at points t."""
def build_vector(t):
    t = np.asarray(t, dtype=float)
    return np.stack([basis(k, t) for k in range(6)], axis=-1)

"""Min–max normalize with safe division."""
def norm(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (X - mins) / denom
    return U, mins, maxs

"""Build the 3D tensor-product Bernstein design matrix"""
def build_bernstein_3d(U):

    U = np.asarray(U, dtype=float)

    x, y, z = U[:, 0], U[:, 1], U[:, 2]

    bx = build_vector(x)
    by = build_vector(y)
    bz = build_vector(z)

    # multiply and flatten back
    Phi = (bx[:, :, None, None] * by[:, None, :, None] * bz[:, None, None, :]).reshape(U.shape[0], -1)
    return Phi.astype(float)

""" Fit a degree-5 Bernstein by least squares."""
def fit_bernstein_3d(actual, corrected):
    X = np.asarray(actual, float)
    Y = np.asarray(corrected, float)

    Y_flat = Y.reshape(-1, 3)
    X_flat = X.reshape(-1, 3)

    U, mins, maxs = norm(X_flat)

    # Build design matrix, solve least squares
    Phi = build_bernstein_3d(U) # (3375, 216)
    coeff, resid, rank, s = LA.lstsq(Phi, Y_flat, rcond=None)

    Ax = coeff[:, 0].reshape(6, 6, 6)
    Ay = coeff[:, 1].reshape(6, 6, 6)
    Az = coeff[:, 2].reshape(6, 6, 6)
    return dict(Ax=Ax, Ay=Ay, Az=Az, mins=mins, maxs=maxs)

"""Apply a fitted Bernstein model to new 3D points."""
def bernstein_out(model, measured_xyz):
    Q = np.asarray(measured_xyz, float)

    mins, maxs = model['mins'], model['maxs']
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (Q - mins) / denom

    bx, by, bz = build_vector(U[:, 0]), build_vector(U[:, 1]) , build_vector(U[:, 2])

    Ax, Ay, Az = model['Ax'], model['Ay'], model['Az']
    N = Q.shape[0]
    out = np.empty((N, 3), float)

    for n in range(N):
        sx = sy = sz = 0.0
        for i in range(6):
            for j in range(6):
                vx = vy = vz = 0.0
                for k in range(6):
                    bzk = bz[n, k]
                    vx += Ax[i, j, k] * bzk
                    vy += Ay[i, j, k] * bzk
                    vz += Az[i, j, k] * bzk
                bxy = bx[n, i] * by[n, j]
                sx += bxy * vx
                sy += bxy * vy
                sz += bxy * vz
        out[n] = (sx, sy, sz)
    return out
