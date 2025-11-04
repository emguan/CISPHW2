"""
Bernstein 5th order polynomial fitting package

Author: Emily Guan
"""

import numpy as np
import numpy.linalg as LA
import math

def coefficients(k):
    return math.comb(5, k)

def basis(k, t):
    t = np.asarray(t, dtype=float)
    return coefficients(k) * (t ** k) * ((1 - t) ** (5 - k))

def build_vector(t):
    t = np.asarray(t, dtype=float)
    return np.stack([basis(k, t) for k in range(6)], axis=-1)

def norm(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (X - mins) / denom
    return U, mins, maxs

def build_bernstein_3d(U):
    """
    U: (N, 3) array of normalized xyz in [0,1]
    Returns: (N, 216) design matrix whose columns are B_i(x)*B_j(y)*B_k(z), i,j,k=0..5
    """
    U = np.asarray(U, dtype=float)
    assert U.ndim == 2 and U.shape[1] == 3, "U must be (N,3)"
    x, y, z = U[:, 0], U[:, 1], U[:, 2]

    bx = build_vector(x)  # (N, 6)
    by = build_vector(y)  # (N, 6)
    bz = build_vector(z)  # (N, 6)

    # Broadcast multiply to (N, 6, 6, 6) then flatten last three dims to 216
    Phi = (bx[:, :, None, None] * by[:, None, :, None] * bz[:, None, None, :]).reshape(U.shape[0], -1)
    return Phi.astype(float)


def fit_bernstein_3d(actual, corrected):
    X = np.asarray(actual, float)        # (F, P, 3) = (125, 27, 3)
    Y = np.asarray(corrected, float)     # shape could be (F, P, 3) or already flattened

    # Flatten to (N,3)
    if Y.ndim == 3:
        Y_flat = Y.reshape(-1, 3)
    else:
        Y_flat = Y  # e.g., if it's already (N,3)
    X_flat = X.reshape(-1, 3)

    # Normalize on all points jointly
    U, mins, maxs = norm(X_flat)         # each is (3,)

    # Build design matrix (N, 216) and solve least squares
    Phi = build_bernstein_3d(U)          # (3375, 216) for your data
    coeff, resid, rank, s = LA.lstsq(Phi, Y_flat, rcond=None)  # coeff: (216, 3)

    Ax = coeff[:, 0].reshape(6, 6, 6)
    Ay = coeff[:, 1].reshape(6, 6, 6)
    Az = coeff[:, 2].reshape(6, 6, 6)
    return dict(Ax=Ax, Ay=Ay, Az=Az, mins=mins, maxs=maxs)


def bernstein_out(model, measured_xyz):
    Q = np.asarray(measured_xyz, float)

    mins, maxs = model['mins'], model['maxs']
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (Q - mins) / denom

    bx = build_vector(U[:, 0]) 
    by = build_vector(U[:, 1]) 
    bz = build_vector(U[:, 2])

    Ax, Ay, Az = model['Ax'], model['Ay'], model['Az']  # (6,6,6)
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
