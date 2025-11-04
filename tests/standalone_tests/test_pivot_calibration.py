# python -m tests.standalone_tests.test_pivot_calibration

import numpy as np
from utils.calibration.calibrate_pivot import pivot_calibration
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points

def _rand_rot(rng):
    M = rng.normal(size=(3,3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

def _rms(A): 
    return float(np.sqrt(np.mean(np.sum(A*A, axis=-1))))

def test_pivot_core_math(pivot_calibration):
    rng = np.random.default_rng(0)
    p_tip_true = np.array([20.0, -15.0, 35.0])
    d_true     = np.array([100.0, 120.0, 80.0])

    R_list, t_list = [], []
    for _ in range(40):
        Rk = _rand_rot(rng)
        tk = d_true - Rk @ p_tip_true   # ensures Rk p + tk = d
        R_list.append(Rk); t_list.append(tk)

    p_tip, d_est, score = pivot_calibration(R_list, t_list)

    assert np.allclose(p_tip, p_tip_true, atol=1e-10)
    assert np.allclose(d_est, d_true,     atol=1e-10)
    # score is LS residual per frame; should be ~0
    assert score < 1e-10


def _rand_rot(rng):
    M = rng.normal(size=(3,3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

def test_pivot_with_registration(pivot_calibration):
    rng = np.random.default_rng(1)
    # A plausible 6-marker probe layout
    B0 = np.array([
        [  0.,  0.,  0.],
        [ 30.,  0.,  0.],
        [  0., 30.,  0.],
        [  0.,  0., 30.],
        [ 30., 30.,  0.],
        [ 30.,  0., 30.],
    ], float)

    p_tip_true = np.array([18., -12., 26.])
    d_true     = np.array([105.,  98., 75.])

    R_list, t_list = [], []
    for _ in range(40):
        Rk = _rand_rot(rng)
        tk = d_true - Rk @ p_tip_true
        Mk = (B0 @ Rk.T) + tk  # synthesize markers in base (row-vector form)

        T  = aruns_method(arr_to_points(B0), arr_to_points(Mk))
        Rh = np.asarray(T.r.R, float)
        th = np.asarray(T.p.points_3d(), float)
        R_list.append(Rh); t_list.append(th)

    p_tip, d_est, score = pivot_calibration(R_list, t_list)

    assert np.allclose(p_tip, p_tip_true, atol=1e-6)
    assert np.allclose(d_est, d_true,     atol=1e-6)
    assert score < 1e-6

if __name__ == "__main__":
    test_pivot_core_math(pivot_calibration)
    test_pivot_with_registration(pivot_calibration)