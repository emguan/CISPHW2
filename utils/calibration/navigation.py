import numpy as np
from utils.IO.read import read_emnav
from utils.mathpackage.bernstein import bernstein_out
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points, frames_to_array

def _extract_R_t_from_F(F_reg):
    """
    Accept either your Transformations wrapper or raw (R,t) and return (R,t) as arrays.
    """
    # Transformations(.r is Rotations, .p is Points)
    if hasattr(F_reg, "r") and hasattr(F_reg, "p"):
        # Rotations may store as .R; Points via points_3d()
        R = np.asarray(getattr(F_reg.r, "R", F_reg.r), float)
        t = np.asarray(F_reg.p.points_3d(), float)
        return R, t
    # Tuple or dict or raw
    if isinstance(F_reg, tuple) and len(F_reg) == 2:
        R, t = F_reg
        return np.asarray(R, float), np.asarray(t, float)
    if isinstance(F_reg, dict) and "R" in F_reg and "t" in F_reg:
        return np.asarray(F_reg["R"], float), np.asarray(F_reg["t"], float)
    # Assume already arrays (R,t)
    return np.asarray(F_reg[0], float), np.asarray(F_reg[1], float)

def compute_nav_tip_ct(model, emnav_file, B0_points, b_tip, F_reg):

    # 1) Read and distortion-correct all nav frames
    G_frames, NB, Nf = read_emnav(emnav_file)               # frames of 6 markers
    raw = frames_to_array(G_frames)                         # (Nf, NB, 3)
    corr = bernstein_out(model, raw.reshape(-1, 3)).reshape(raw.shape)

    # 2) For each frame: register to B0, compute tip in EM base
    tips_em = np.empty((Nf, 3), float)
    for k in range(Nf):
        T = aruns_method(B0_points, arr_to_points(corr[k])) # body -> EM base
        Rk = np.asarray(T.r.R, float)
        tk = np.asarray(T.p.points_3d(), float)
        tips_em[k] = b_tip @ Rk.T + tk

    # 3) Apply F_reg (EM->CT)
    R_reg, t_reg = F_reg.r.R, F_reg.p.points_3d()
    #print(R_reg)
    tips_ct = (tips_em @ R_reg.T) + t_reg

    return tips_em, tips_ct
