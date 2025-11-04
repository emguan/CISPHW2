"""
Main workflow as described in PA2 hand out. 

Example execution: 
python -m scripts.distortion.workflow --calbody data/pa2-unknown-j-calbody.txt --calreadings data/pa2-unknown-j-calreadings.txt --empivot data/pa2-unknown-j-empivot.txt --emfiducials data/pa2-unknown-j-em-fiducialss.txt --ctfiducials data/pa2-unknown-j-ct-fiducials.txt --emnav data/pa2-unknown-j-EM-nav.txt --output pa2-unknown-j-EM-output2.txt
python -m scripts.distortion.workflow --calbody data/pa2-debug-f-calbody.txt --calreadings data/pa2-debug-f-calreadings.txt --empivot data/pa2-debug-f-empivot.txt --emfiducials data/pa2-debug-f-em-fiducialss.txt --ctfiducials data/pa2-debug-f-ct-fiducials.txt --emnav data/pa2-debug-f-EM-nav.txt --output pa2-debug-f-EM-output2.txt

Author: Emily Guan
"""

import argparse
import numpy as np

from tests.runtime_tests.test_model import check_model

from utils.calibration.expected import calibrate
from utils.mathpackage.mathpackage import Points, Rotations, Transformations
from utils.IO.write import write_expected_2

# step 1
from utils.IO.read import read_ctfiducials

# step 2
from utils.mathpackage.bernstein import fit_bernstein_3d, bernstein_out

# step 3
from utils.calibration.calibrate_pivot import calibrate_pivot

# step 4
from utils.calibration.calc_fids import calc_emfid

# step 5
from utils.calibration.calc_reg_frame import register_em_to_ct

# step 6
from utils.calibration.navigation import compute_nav_tip_ct

"""
Main executable for all of homework, using command line file inputs.
"""
def main(calbody_file, calreadings_file, ctfid_file, emfid_file, emnav_file, empivot_file, outfile):

    # step 1: finding C_expected
    C_actual, C_pred = calibrate(calbody_file, calreadings_file)

    C_actual = np.asarray(C_actual, dtype=float)
    C_pred = np.asarray(C_pred, dtype=float)

    C_actual = C_actual.reshape(-1, 3)
    C_pred = C_pred.reshape(-1, 3)

    errs = np.linalg.norm(C_pred - C_actual, axis=1) # calculating mean error, max error, and number of errors
    print(f"[calibrate] mean={errs.mean():.6f}, max={errs.max():.6f}, n={len(errs)}")

    # step 2: fitting distortion model
    model = fit_bernstein_3d(C_actual, C_pred)
    
    check_model(model) # checking model fits dimensional requirements

    # step 3: pivot calibration w.r.t. EM probe using distortion function
    b_tip, b_post, score, B0, R_list, t_list = calibrate_pivot(model, empivot_file)
    print(f"[pivot] score={score:.6f}, ||b_tip||={np.linalg.norm(b_tip):.3f}")

    # step 4: calculate location of fiducials w.r.t. EM 
    b = calc_emfid(model, emfid_file, B0, b_tip)
    print("[em fids] first 3:\n", np.array2string(b[:3], precision=3))

    # step 5: compute registration frame
    F_em2ct, F_ct2em, rms = register_em_to_ct(b, ctfid_file)
    print("[reg] rms:", rms)

    # --- Step 6: navigation ---
    tips_em, tips_ct = compute_nav_tip_ct(model, emnav_file, B0, b_tip, F_em2ct)
    d = np.linalg.norm(tips_ct - tips_em, axis=1)
    print(f"[nav] mean |CT-EM|={d.mean():.3f}, max={d.max():.3f}")

    # write ct results using format stated in HW handout
    write_expected_2(outfile, tips_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA2 pipeline: distortion, pivot, fiducials, registration, nav → CT")
    parser.add_argument("--calbody", required=True)
    parser.add_argument("--calreadings", required=True)
    parser.add_argument("--empivot", required=True)
    parser.add_argument("--emfiducials", required=True)
    parser.add_argument("--ctfiducials", required=True)
    parser.add_argument("--emnav", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.calbody, args.calreadings, args.ctfiducials, args.emfiducials, args.emnav, args.empivot, args.output)
