import numpy as np
from utils.mathpackage.rigid_transform import aruns_method  # returns Transformations(.r, .p)
from utils.helpers.converters import arr_to_points          # (N,3) -> list[Points]
from utils.IO.read import read_ctfiducials
from utils.mathpackage.mathpackage import Transformations, Rotations, Points

def register_em_to_ct(B_em: np.ndarray, ct_file: str):
    """
    Compute EM->CT registration using matched fiducials.

    Args:
      B_em   : (M,3) EM-base fiducials (averaged per physical fid)
      ct_file: path to CT fiducials file

    Returns:
      R_em2ct, t_em2ct, R_ct2em, t_ct2em, rms
    """
    # Load CT fiducials (list[Points]) and count
    ct_pts, Nc = read_ctfiducials(ct_file)   # adjust if your read returns only the list

    # Convert CT fiducials to ndarray for error calc
    C_ct = np.array([p.points_3d() for p in ct_pts], dtype=float)  # (M,3)

    if B_em.shape != C_ct.shape or B_em.shape[1] != 3:
        raise ValueError(f"Shape mismatch: B_em {B_em.shape} vs C_ct {C_ct.shape}. Need (M,3) each in same order.")

    # Arun/Kabsch expects list[Points]; B_em is ndarray, CT is already Points list
    T = aruns_method(arr_to_points(B_em), ct_pts)  # C ≈ R B + t

    R_em2ct = np.asarray(T.r.R, dtype=float)           # (3,3)
    t_em2ct = np.asarray(T.p.points_3d(), dtype=float) # (3,)

    # Inverse (CT -> EM)
    R_ct2em = R_em2ct.T
    t_ct2em = -R_ct2em @ t_em2ct

    # RMS registration error
    C_pred = (B_em @ R_em2ct.T) + t_em2ct
    errs = np.linalg.norm(C_pred - C_ct, axis=1)
    rms = float(np.sqrt(np.mean(errs**2)))

    F_em2ct = Transformations("em to ct", Rotations(R_em2ct), Points(t_em2ct))

    F_ct2em = Transformations("ct to em", Rotations(R_ct2em), Points(t_ct2em))
    print(rms) # 0.0048232900879437764

    return F_em2ct ,F_ct2em, rms
