from utils.mathpackage.bernstein import bernstein_out
from utils.IO.read import read_emfiducials 

def calibrate_pivot(model, emfid_path):

    B_frames, NB, Nframes = read_emfiducials(emfid_path)

    B_list = []

    for frame in B_frames:
        B_list.append([float(p.x), float(p.y), float(p.z)])

    out = bernstein_out(model, B_frames)

    return out

def compute_b(delta_b, ):
    return