"""
Given file path, will write files for easy use according to formats stated in hw document.

Author: Emily Guan
"""

from utils.mathpackage.mathpackage import Points

import numpy as np

"""
Given sample name, will write out expected values. 
"""
def write_expected_1(name: str, C_pred_list: list):
    NC = C_pred_list[0].shape[0]  
    Nframes = len(C_pred_list)
    path = "./output1/" + name
    with open(path, "w") as f:
        f.write(f"{NC}, {Nframes}, {name}\n")
        for k, Ck in enumerate(C_pred_list):
            for row in Ck:
                f.write(f"{row[0]:.2f},{row[1]:.2f},{row[2]:.2f}\n")

"""
Given sample name, will write out probe tip in CT. 
"""
def write_expected_2(name: str, C_pred_list: list):
    NC = C_pred_list[0].shape[0]  
    Nframes = len(C_pred_list)
    path = "./output2/" + name
    with open(path, "w") as f:
        f.write(f"{Nframes}, {name}\n")
        for k, row in enumerate(C_pred_list):
            print(row)
            f.write(f"{row[0]:.2f},{row[1]:.2f},{row[2]:.2f}\n")


                