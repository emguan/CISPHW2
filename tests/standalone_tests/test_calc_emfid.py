# python -m tests.standalone_tests.test_calc_emfid
import numpy as np
import math
import sys

# --- imports from your repo (adjust paths if needed) ---
from utils.mathpackage.bernstein import fit_bernstein_3d, bernstein_out, build_vector
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points
from utils.calibration.calibrate_pivot import pivot_calibration
import utils.calibration.calc_fids as calc_fids_mod  # we'll temporarily override functions in here

# --------------- helpers ---------------
def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(np.sum(a*a, axis=-1))))

def rand_rot(rng):
    M = rng.normal(size=(3,3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1
    return Q

def einsum_eval(model, Q):
    Q = np.asarray(Q, float)
    mins, maxs = model["mins"], model["maxs"]
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    U = (Q - mins)/denom
    Ax,Ay,Az = model["Ax"],model["Ay"],model["Az"]
    bx = build_vector(U[:,0])  # (N,6)
    by = build_vector(U[:,1])
    bz = build_vector(U[:,2])
    sx = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ax)
    sy = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Ay)
    sz = np.einsum("ni,nj,nk,ijk->n", bx, by, bz, Az)
    return np.stack([sx,sy,sz], axis=1)

def make_probe_template():
    return np.array([
        [  0.,  0.,  0.],
        [ 30.,  0.,  0.],
        [  0., 30.,  0.],
        [  0.,  0., 30.],
        [ 30., 30.,  0.],
        [ 30.,  0., 30.],
    ], float)

# --------------- tests ---------------
def test_bernstein_identity():
    print("\n[bernstein] identity mapping...")
    rng = np.random.default_rng(0)
    X = rng.uniform([-100,-100,0],[100,100,200], size=(800,3))
    Y = X.copy()
    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    e = rms(Yhat - Y)
    print(f"  RMS: {e:.3e}")
    assert e < 1e-8

def test_bernstein_affine():
    print("[bernstein] affine mapping...")
    rng = np.random.default_rng(1)
    X = rng.uniform([-150,-120,50], [140,130,180], size=(2000,3))
    R = rand_rot(rng)
    t = np.array([5.0, -12.0, 3.0])
    Y = (X @ R.T) + t
    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    Yhat = bernstein_out(model, X)
    e = rms(Yhat - Y)
    print(f"  RMS: {e:.3e}")
    assert e < 1e-8

def test_bernstein_fast_equals_einsum():
    print("[bernstein] fast vs einsum evaluator...")
    rng = np.random.default_rng(2)
    X = rng.uniform([-80,-60,20],[90,70,160], size=(1200,3))
    # mild nonlinearity
    Y = X.copy()
    Y[:,0] += 1e-3 * (X[:,1]*X[:,2])/1e3
    Y[:,1] += 1e-3 * (X[:,0]*X[:,2])/1e3
    Y[:,2] += 1e-3 * (X[:,0]*X[:,1])/1e3
    model = fit_bernstein_3d(X.reshape(-1,1,3), Y.reshape(-1,1,3))
    fast = bernstein_out(model, X)
    ref  = einsum_eval(model, X)
    diff = np.max(np.abs(fast - ref))
    print(f"  max|fast-ref|: {diff:.3e}")
    assert diff < 1e-10

def test_pivot_core():
    print("\n[pivot] core LS solve...")
    rng = np.random.default_rng(3)
    p_tip_true = np.array([20.0, -15.0, 35.0])
    d_true     = np.array([100.0, 120.0, 80.0])
    R_list, t_list = [], []
    for _ in range(40):
        Rk = rand_rot(rng)
        tk = d_true - Rk @ p_tip_true
        R_list.append(Rk); t_list.append(tk)
    p_tip, d_est, score = pivot_calibration(R_list, t_list)
    print(f"  ||p_tip-p_true||: {np.linalg.norm(p_tip-p_tip_true):.3e}")
    print(f"  ||d-d_true||:     {np.linalg.norm(d_est-d_true):.3e}")
    print(f"  LS score:         {score:.3e}")
    assert np.allclose(p_tip, p_tip_true, atol=1e-10)
    assert np.allclose(d_est, d_true,     atol=1e-10)
    assert score < 1e-10

def test_pivot_with_registration():
    print("[pivot] with registration (aruns_method)...")
    rng = np.random.default_rng(4)
    B0 = make_probe_template()
    p_tip_true = np.array([18., -12., 26.])
    d_true     = np.array([105.,  98., 75.])
    R_list, t_list = [], []
    for _ in range(40):
        Rk = rand_rot(rng)
        tk = d_true - Rk @ p_tip_true
        Mk = (B0 @ Rk.T) + tk
        T  = aruns_method(arr_to_points(B0), arr_to_points(Mk))
        Rh = np.asarray(T.r.R, float)
        th = np.asarray(T.p.points_3d(), float)
        R_list.append(Rh); t_list.append(th)
    p_tip, d_est, score = pivot_calibration(R_list, t_list)
    print(f"  ||p_tip-p_true||: {np.linalg.norm(p_tip-p_tip_true):.3e}")
    print(f"  ||d-d_true||:     {np.linalg.norm(d_est-d_true):.3e}")
    print(f"  LS score:         {score:.3e}")
    assert np.allclose(p_tip, p_tip_true, atol=1e-6)
    assert np.allclose(d_est, d_true,     atol=1e-6)
    assert score < 1e-6

def test_calc_emfid_no_files():
    """
    Test calc_emfid by temporarily replacing read_emfiducials + bernstein_out
    with synthetic generators / identity undistortion. No pytest needed.
    """
    print("\n[calc_emfid] synthetic frames, no file I/O...")
    # save originals to restore later
    orig_reader = calc_fids_mod.read_emfiducials
    orig_bern   = calc_fids_mod.bernstein_out
    try:
        # build synthetic frames
        B0 = make_probe_template()
        p_tip_true = np.array([14., -5., 22.], float)
        d_true     = np.array([100.,130., 80.], float)
        rng = np.random.default_rng(5)
        Nf = 30
        frames = []
        for _ in range(Nf):
            Rk = rand_rot(rng)
            tk = d_true - Rk @ p_tip_true
            Mk = (B0 @ Rk.T) + tk + rng.normal(scale=0.1, size=(B0.shape[0],3))
            frames.append([tuple(Mk[i]) for i in range(B0.shape[0])])
        NB = B0.shape[0]

        # override reader
        def fake_read_emfiducials(_path):
            return frames, NB, Nf
        calc_fids_mod.read_emfiducials = fake_read_emfiducials

        # identity distortion
        def id_bern(_model, Q): return np.asarray(Q, float)
        calc_fids_mod.bernstein_out = id_bern

        tips = calc_fids_mod.calc_emfid(model={}, emfid_file="<ignored>", B0=B0, b_tip=p_tip_true)
        err = rms(tips - d_true[None,:])
        print(f"  tips shape: {tips.shape}, RMS to d_true: {err:.3e}")
        assert tips.shape == (Nf, 3)
        assert err < 0.5  # loose bound due to registration + noise
    finally:
        # restore
        calc_fids_mod.read_emfiducials = orig_reader
        calc_fids_mod.bernstein_out    = orig_bern

# --------------- main ---------------
def main():
    try:
        test_bernstein_identity()
        test_bernstein_affine()
        test_bernstein_fast_equals_einsum()
        test_pivot_core()
        test_pivot_with_registration()
        test_calc_emfid_no_files()
        print("\nALL SANITY CHECKS PASSED ✅")
    except AssertionError as e:
        print("\n❌ Test failed:", e)
        sys.exit(1)
    except Exception as e:
        print("\n💥 Error running tests:", repr(e))
        sys.exit(2)

if __name__ == "__main__":
    main()
