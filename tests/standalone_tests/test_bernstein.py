# python -m tests.standalone_tests.test_bernstein
import numpy as np
import pickle
import math

# Adjust imports if your path differs
from utils.mathpackage.bernstein import fit_bernstein_3d, bernstein_out, build_vector

# ---------- helpers ----------

def rms(err):
    err = np.asarray(err, float)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))

def rand_rot(rng):
    # random proper rotation via QR
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def eval_einsum(model, Q):
    """
    Reference evaluator using einsum + basis vectors, independent of bernstein_out loop.
    """
    Q = np.asarray(Q, float)
    mins, maxs = model["mins"], model["maxs"]
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (Q - mins) / denom

    Ax, Ay, Az = model["Ax"], model["Ay"], model["Az"]
    bx = build_vector(U[:, 0])  # (N,6)
    by = build_vector(U[:, 1])  # (N,6)
    bz = build_vector(U[:, 2])  # (N,6)

    # For each axis, compute sum_{i,j,k} Ax[i,j,k]*bx_i*by_j*bz_k
    sx = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ax)
    sy = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ay)
    sz = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Az)
    return np.stack([sx, sy, sz], axis=1)

# ---------- tests ----------

def test_model_shapes_after_fit():
    rng = np.random.default_rng(0)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(200,3))
    # simple identity target
    Y = X.copy()

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    assert isinstance(model, dict)
    for k in ["Ax", "Ay", "Az"]:
        assert k in model and model[k].shape == (6,6,6)
    for k in ["mins", "maxs"]:
        assert k in model and model[k].shape == (3,)
    assert np.all(model["maxs"] >= model["mins"])

def test_fit_predict_identity_low_error():
    rng = np.random.default_rng(1)
    X = rng.uniform([-100,-100,0], [100,100,200], size=(500,3))
    Y = X.copy()

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    assert rms(Yhat - Y) < 1e-8  # essentially perfect for identity

def test_fit_predict_affine_low_error():
    rng = np.random.default_rng(2)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(2000,3))
    R = rand_rot(rng)
    t = np.array([5.0, -12.0, 3.0])
    Y = (X @ R.T) + t  # row-vector convention

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    assert rms(Yhat - Y) < 1e-8  # 5th-order should nail an affine map

def test_fast_equals_einsum_reference():
    rng = np.random.default_rng(3)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(1500,3))
    # Use some gentle nonlinear target to avoid degenerate affine-only test
    # e.g., add small 2nd-order term
    Y = X.copy()
    Y[:,0] += 1e-3 * (X[:,1]*X[:,2]) / 1e3
    Y[:,1] += 1e-3 * (X[:,0]*X[:,2]) / 1e3
    Y[:,2] += 1e-3 * (X[:,0]*X[:,1]) / 1e3

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    fast = bernstein_out(model, X)
    ref  = eval_einsum(model, X)
    assert np.allclose(fast, ref, atol=1e-10, rtol=0)

def test_batching_and_slicing_invariance():
    rng = np.random.default_rng(4)
    X = rng.uniform([-50,-50,0], [50,50,100], size=(400,3))
    Y = X + np.array([1.0, -2.0, 3.5])  # simple shift

    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    full = bernstein_out(model, X)
    half = bernstein_out(model, X[::2])
    assert np.allclose(full[::2], half)

def test_extrapolation_runs_and_is_finite():
    rng = np.random.default_rng(5)
    X = rng.uniform([0,0,0], [100,100,100], size=(500,3))
    Y = X + 2.0  # shift
    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))

    # Query well outside the training box
    Q = rng.uniform([200,200,200], [300,300,300], size=(50,3))
    out = bernstein_out(model, Q)
    assert np.all(np.isfinite(out))  # no NaNs or infs

def test_pickle_roundtrip_model():
    rng = np.random.default_rng(6)
    X = rng.uniform([-100,-80,20], [90,70,160], size=(300,3))
    Y = 0.5 * X + np.array([10, -5, 7])
    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))

    blob = pickle.dumps(model)
    model2 = pickle.loads(blob)

    Xq = rng.uniform([-100,-80,20], [90,70,160], size=(20,3))
    out1 = bernstein_out(model,  Xq)
    out2 = bernstein_out(model2, Xq)
    assert np.allclose(out1, out2)

if __name__ == "__main__":
    test_model_shapes_after_fit()
    test_fit_predict_identity_low_error()
    test_fit_predict_affine_low_error()
    test_fast_equals_einsum_reference()
    test_batching_and_slicing_invariance()
    test_extrapolation_runs_and_is_finite()
    test_pickle_roundtrip_model()



