"""
Testing Bernstein math package.

Executable:
python -m tests.standalone_tests.test_bernstein

Author: Emily Guan
"""

import numpy as np
import pickle
import math

from utils.mathpackage.bernstein import fit_bernstein_3d, bernstein_out, build_vector

#----------helpers----------

def rms(err): # calculate rms error
    err = np.asarray(err, float)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))

def rand_rot(rng): # random proper rotation via QR
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

"""
The premade alternative to the 4-nested loop that I wrote in bernstein_out 
can be tested/replaced with einsum. 

https://numpy.org/doc/stable/reference/generated/numpy.einsum.html 
"""
def eval_einsum(model, Q):

    Q = np.asarray(Q, float)
    mins, maxs = model["mins"], model["maxs"]
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (Q - mins) / denom

    Ax, Ay, Az = model["Ax"], model["Ay"], model["Az"]
    bx = build_vector(U[:, 0])
    by = build_vector(U[:, 1])
    bz = build_vector(U[:, 2])

    # sum_{i,j,k} Ax[i,j,k]*bx_i*by_j*bz_k
    sx = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ax)
    sy = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ay)
    sz = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Az)
    return np.stack([sx, sy, sz], axis=1)

#----------tests----------

# test that an identity transformation leads to no problems
def test_fit_predict_identity():
    rng = np.random.default_rng(1)
    X = rng.uniform([-100,-100,0], [100,100,200], size=(500,3))
    Y = X.copy() #identity

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    assert rms(Yhat - Y) < 1e-8  # essentially perfect for identity

# test that after a random transformation, bernstein prediction is still similar to original
def test_fit_predict_transformation():
    rng = np.random.default_rng(2)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(2000,3))
    R = rand_rot(rng) # random rotation
    t = np.array([5.0, -12.0, 3.0])
    Y = (X @ R.T) + t  # row-vector convention

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    assert rms(Yhat - Y) < 1e-8

#testing my 4 nested for loop works same as einsum
def test_fast_equals_einsum_reference():
    rng = np.random.default_rng(3)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(1500,3))
    Y = X.copy()

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    fast = bernstein_out(model, X)
    ref  = eval_einsum(model, X)
    assert np.allclose(fast, ref, atol=1e-10, rtol=0)

if __name__ == "__main__":
    test_fit_predict_identity()
    test_fit_predict_transformation()
    test_fast_equals_einsum_reference()


