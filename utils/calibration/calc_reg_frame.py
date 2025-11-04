"""
Calculating registration frame for step 5 of workflow.

Author: Emily Guan
"""

import numpy as np

from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points
from utils.IO.read import read_ctfiducials
from utils.mathpackage.mathpackage import Transformations, Rotations, Points

"""
Registers EM fiducials to CT fiducials read from ctfid_file.
Returns (F_em2ct, F_ct2em, rms) where rms is computed on those two sets.
"""
def register_em_to_ct(B_em: np.ndarray, ctfid_file: str):

    B_em = np.asarray(B_em, float)
    assert B_em.ndim == 2 and B_em.shape[1] == 3, f"B_em bad shape {B_em.shape}"

    ct_read = read_ctfiducials(ctfid_file)
    C_ct = np.array([p.points_3d() for p in ct_pts], dtype=float)
    assert C_ct.shape == B_em.shape, f"Shape mismatch: B_em {B_em.shape} vs C_ct {C_ct.shape}"

    T = aruns_method(arr_to_points(B_em), arr_to_points(C_ct))
    R = np.asarray(T.r.R, dtype=float)
    t = np.asarray(T.p.points_3d(), dtype=float).reshape(3)

    # transforms
    F_em2ct = Transformations("em to ct", Rotations(R), Points("pt", t[0], t[1],t[2]))
    R_inv = R.T
    t_inv = -R_inv @ t
    F_ct2em = Transformations("ct to em", Rotations(R_inv), Points("pt", t_inv[0], t_inv[1], t_inv[2])) # backwards ct2em is reverse of forward em2ct

    # rms
    C_pred = (B_em @ R.T) + t
    residuals = np.linalg.norm(C_pred - C_ct, axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))

    """print("[reg] t:", t, "rms:", rms)
    print(f"[reg] residuals: min={residuals.min():.4f}, med={np.median(residuals):.4f}, max={residuals.max():.4f}") 
    """
    
    return F_em2ct, F_ct2em, rms
