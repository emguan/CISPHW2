"""
Reads in EMNAV data and 
"""

import numpy as np
from utils.IO.read import read_emnav
from utils.mathpackage.bernstein import bernstein_out
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points, frames_to_array

"""
Takes Transformation and breaks it down into R and p.
"""
def _extract_R_t_from_F(F_reg):
    
    R = np.asarray(getattr(F_reg.r, "R", F_reg.r), float)
    t = np.asarray(F_reg.p.points_3d(), float)
    return R, t

"""
Compute tip w.r.t CT image for step 6. 
"""
def compute_nav_tip_ct(model, emnav_file, B0_points, b_tip, F_reg):

    G_frames, NB, Nf = read_emnav(emnav_file) 
    raw = frames_to_array(G_frames) 
    corr = bernstein_out(model, raw.reshape(-1, 3)).reshape(raw.shape)

    # register to B0, compute tip in EM base
    tips_em = np.empty((Nf, 3), float)
    for k in range(Nf):
        T = aruns_method(B0_points, arr_to_points(corr[k])) # body -> EM base
        Rk = np.asarray(T.r.R, float)
        tk = np.asarray(T.p.points_3d(), float)
        tips_em[k] = b_tip @ Rk.T + tk

    # F_reg (EM->CT)
    R_reg, t_reg = F_reg.r.R, F_reg.p.points_3d()
    tips_ct = (tips_em @ R_reg.T) + t_reg

    return tips_em, tips_ct
