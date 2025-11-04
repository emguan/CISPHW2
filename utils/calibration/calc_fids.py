"""
Calculating EM fiducial locations for step 4 of workflow.

Author: Emily Guan
"""

import numpy as np

from utils.mathpackage.bernstein import bernstein_out
from utils.IO.read import read_emfiducials
from utils.mathpackage.mathpackage import Points
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points, frames_to_array

"""Compute per-frame EM tool-tip positions by calibrating raw fiducials, 
registering each frame to base body B0 via Arun’s method, 
and transforming the body-frame tip b_tip into EM space."""
def calc_emfid(model, emfid_file, B0, b_tip):

    frames, NB, Nf = read_emfiducials(emfid_file)
    raw = frames_to_array(frames)

    corr = bernstein_out(model, raw.reshape(-1, 3)).reshape(raw.shape)

    tips = np.empty((Nf, 3))
    for k in range(Nf):
        Tk = aruns_method(B0, arr_to_points(corr[k]))
        Rk = np.asarray(Tk.r.R)
        tk = np.asarray(Tk.p.points_3d())
        tips[k] = b_tip @ Rk.T + tk

    return tips