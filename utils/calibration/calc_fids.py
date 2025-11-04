import numpy as np

from utils.mathpackage.bernstein import bernstein_out
from utils.IO.read import read_emfiducials
from utils.mathpackage.mathpackage import Points
from utils.mathpackage.rigid_transform import aruns_method
from utils.helpers.converters import arr_to_points, frames_to_array

def calc_emfid(model, emfid_file, B0, b_tip):

    B_frames, NB, Nframes = read_emfiducials(emfid_file)

    frames, NB, Nf = read_emfiducials(emfid_file)

    # 1) Distortion-correct all marker observations
    raw = frames_to_array(frames)                                 # (Nf, NB, 3)
    corr = bernstein_out(model, raw.reshape(-1, 3)).reshape(raw.shape)

    # 2) For each frame: register to B0, then transform the tip
    tips = np.empty((Nf, 3), dtype=float)
    for k in range(Nf):
        Tk = aruns_method(B0, arr_to_points(corr[k]))       # body -> EM base
        Rk = np.asarray(Tk.r.R, float)                              # (3,3)
        tk = np.asarray(Tk.p.points_3d(), float)                    # (3,)
        tips[k] = Rk @ b_tip + tk

    return tips