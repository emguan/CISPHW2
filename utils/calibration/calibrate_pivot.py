import numpy as np
from utils.mathpackage.bernstein import bernstein_out
from utils.IO.read import read_empivot
from utils.mathpackage.rigid_transform import aruns_method  # returns object with .r (3x3), .p (3,)
from utils.helpers.converters import arr_to_points, rot_to_arr, frames_to_array

def distortion_pivot(model, empivot_file):
    """Return distortion-corrected EMPIVOT frames as (Nf, NB, 3)."""
    frames, NB, Nf = read_empivot(empivot_file)
    raw = frames_to_array(frames)                     # (Nf, NB, 3)
    corr = bernstein_out(model, raw.reshape(-1, 3))    # (Nf*NB, 3)
    corr = corr.reshape(raw.shape)                     # (Nf, NB, 3)
    return corr

def pivot_calibration(R_list: list[np.ndarray], t_list: list[np.ndarray]):
    """Solve [R|-I][b_tip;b_post] = -t across frames."""
    pts = len(R_list)
    A = np.zeros((3*pts, 6))
    b = np.zeros((3*pts, 1))
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        A[3*i:3*i+3, 0:3] = R
        A[3*i:3*i+3, 3:6] = -np.eye(3)
        b[3*i:3*i+3, 0]   = -np.asarray(t, float).reshape(3)
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_tip  = x[0:3, 0]
    b_post = x[3:6, 0]
    score = float(np.sqrt(res[0] / pts)) if res.size > 0 else float(np.sqrt(np.mean((A @ x - b)**2)))
    #print(f"RES SCORE PIVOT: {score:.6f}")
    return b_tip, b_post, score

def calibrate_pivot(model, empivot_file):
    
    corr = distortion_pivot(model, empivot_file)       # (Nf, NB, 3)
    Nf, NB, _ = corr.shape

    # Use first corrected frame as probe template B0
    B0 = corr[0].copy()
    B0 = arr_to_points(B0)

    # Register B0 -> each frame to get R_k, t_k
    R_list, t_list = [], []
    for k in range(Nf):
        corrk = arr_to_points(corr[k])
        T = aruns_method(B0, corrk)                  # aligns: corr[k] ≈ R_k B0 + t_k
        R_list.append(T.r.R)
        t_list.append(T.p.points_3d())
    R_list = [np.asarray(R) for R in R_list]
    t_list = [np.asarray(t) for t in t_list]

    # Solve the stacked system
    b_tip, b_post, score = pivot_calibration(R_list, t_list)
    return b_tip, b_post, score, B0, R_list, t_list
