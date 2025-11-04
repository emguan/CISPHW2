"""
Main workflow as described in PA2 hand out. 

Example execution: python -m scripts.distortion.workflow --calbody --calreadings --empivot --emfiducials --ctfiducials --emnav

Author: Emily Guan
"""
import argparse

from scripts.calibration.expected import calibrate
from utils.mathpackage.mathpackage import Points, Rotations, Transformations

# step 1
from utils.IO.read import read_calbody, read_calreadings, read_ctfiducials, read_emfiducials, read_empivot, read_optpivot

# step 2
from utils.mathpackage.bernstein import fit_bernstein_3d

# step 3
from scripts.calibration.calibrate_pivot import calibrate_pivot

def main(calbody_file, calreadings_file, ctfid_file, emfid_file, emnav_file, empivot_file):

    # step 1: find c_exp
    C_actual, C_pred = calibrate(calbody_file, calreadings_file)

    # step 2: find distortion function
    model = fit_bernstein_3d(C_actual, C_pred)

    # step 3: pivot calibration for EM probe
    pivot_cal = calibrate_pivot(model, emfid_file)

    # step 4: computing locations of fiducials w.r.t to EM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA2 pipeline: distortion, pivot, fiducials, registration, nav → CT")
    parser.add_argument("--calbody", required=True)
    parser.add_argument("--calreadings", required=True)
    parser.add_argument("--empivot", required=False)
    parser.add_argument("--emfiducials", required=True)
    parser.add_argument("--ctfiducials", required=True)
    parser.add_argument("--emnav", required=True)
    args = parser.parse_args()

    main(args.calbody, args.calreadings, args.ctfiducials, args.emfiducials, args.emnav, args.empivot)
    
    



