"""
Main workflow as described in PA2 hand out. 

Example execution: 
python -m scripts.distortion.workflow --calbody data/pa2-debug-a-calbody.txt --calreadings data/pa2-debug-a-calreadings.txt --empivot data/pa2-debug-a-empivot.txt --emfiducials data/pa2-debug-a-em-fiducialss.txt --ctfiducials data/pa2-debug-a-ct-fiducials.txt --emnav data/pa2-debug-a-EM-nav.txt 

Author: Emily Guan
"""
import argparse

from tests.runtime_tests.test_model import check_model
from tests.runtime_tests.test_calibrate import check_pred
from utils.calibration.expected import calibrate
from utils.mathpackage.mathpackage import Points, Rotations, Transformations

# step 1
from utils.IO.read import read_calbody, read_calreadings, read_ctfiducials, read_emfiducials, read_empivot, read_optpivot

# step 2
from utils.mathpackage.bernstein import fit_bernstein_3d

# step 3
from utils.calibration.calibrate_pivot import calibrate_pivot

# step 4
from utils.calibration.calc_fids import calc_emfid

# step 5
from utils.calibration.calc_reg_frame import register_em_to_ct

# step 6
from utils.calibration.navigation import compute_nav_tip_ct

def main(calbody_file, calreadings_file, ctfid_file, emfid_file, emnav_file, empivot_file):

    # step 1: find c_exp - works
    C_actual, C_pred = calibrate(calbody_file, calreadings_file)
    check_pred(C_actual, C_pred)

    # step 2: find distortion function
    model = fit_bernstein_3d(C_actual, C_pred)
    check_model(model)

    # step 3: pivot calibration for EM probe
    b_tip, b_post, score, B0, R_list, t_list = calibrate_pivot(model, empivot_file)

    # step 4: computing locations of fiducials w.r.t to EM
    b = calc_emfid(model, emfid_file, B0, b_tip)

    # step 5: calc registration frame
    F_em2ct ,F_ct2em, rms = register_em_to_ct(b, ctfid_file)

    # step 6: 
    tips_em, tips_ct = compute_nav_tip_ct(model, emnav_file, B0, b_tip,F_em2ct)

    print(tips_em, tips_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA2 pipeline: distortion, pivot, fiducials, registration, nav → CT")
    parser.add_argument("--calbody", required=True)
    parser.add_argument("--calreadings", required=True)
    parser.add_argument("--empivot", required=True)
    parser.add_argument("--emfiducials", required=True)
    parser.add_argument("--ctfiducials", required=True)
    parser.add_argument("--emnav", required=True)
    args = parser.parse_args()

    main(args.calbody, args.calreadings, args.ctfiducials, args.emfiducials, args.emnav, args.empivot)
    
