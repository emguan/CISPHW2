"""
Given file path, will read files for easy use according to formats stated in hw document.

Author: Emily Guan
"""

from utils.mathpackage.mathpackage import Points

import numpy as np

#----------------------------HW1----------------------------

"""
Reads & splices 3 coord line.
"""
def read_xyz(line: str):
    x_str, y_str, z_str = line.strip().split(',')
    return float(x_str), float(y_str), float(z_str)

"""
Reads calbody files and returns d, a, c along with counts.
"""
def read_calbody(filepath):
    d_pts, a_pts, c_pts = [], [], []
    with open(filepath, 'r') as f:

        # header = ND, NA, NC, filename
        parts = [p.strip() for p in f.readline().split(',')]
        ND, NA, NC = int(parts[0]), int(parts[1]), int(parts[2])

        #reading d
        for i in range(ND):
            x, y, z = read_xyz(f.readline())
            d_pts.append(Points("EM Base (model)", x, y, z))

        #reading a
        for i in range(NA):
            x, y, z = read_xyz(f.readline())
            a_pts.append(Points("Calib Object (model)", x, y, z))

        #reading c
        for i in range(NC):
            x, y, z = read_xyz(f.readline())
            c_pts.append(Points("Calib Object EM (model)", x, y, z))

    return d_pts, a_pts, c_pts, ND, NA, NC

"""
Reads calreadings files and returns D, A, C and compares counts with calbody to ensure equality.
"""
def read_calreadings(filepath, ND: int, NA: int, NC: int):
    D_frames, A_frames, C_frames = [], [], []
    with open(filepath, 'r') as f:
        # header = ND, NA, NC, Nframes, filename
        parts = [p.strip() for p in f.readline().split(',')]
        ND2, NA2, NC2, Nframes = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        
        # making sure it maches calbody
        if (ND2 != ND) or (NA2 != NA) or (NC2 != NC):
            raise ValueError("Count mismatch between CALBODY and CALREADINGS.")

        for frame in range(Nframes):
            D_list, A_list, C_list = [], [], []

            #reading D
            for i in range(ND):
                x, y, z = read_xyz(f.readline())
                D_list.append(Points("Sensor (optical) D", x, y, z))

            #reading A
            for i in range(NA):
                x, y, z = read_xyz(f.readline())
                A_list.append(Points("Sensor (optical) A", x, y, z))

            #reading C
            for i in range(NC):
                x, y, z = read_xyz(f.readline())
                C_list.append(Points("Sensor (EM) C", x, y, z))

            D_frames.append(D_list)
            A_frames.append(A_list)
            C_frames.append(C_list)

    return D_frames, A_frames, C_frames, Nframes

"""
Reads empivot files and returns G frames and count.
"""
def read_empivot(filepath):
    G_frames = []
    with open(filepath, 'r') as f:
        # header = NG, Nframes, filename
        parts = [p.strip() for p in f.readline().split(',')]
        NG, Nframes = int(parts[0]), int(parts[1])

        for frame in range(Nframes):
            G_list = []
            for i in range(NG):
                x, y, z = read_xyz(f.readline())
                G_list.append(Points("EM Probe (sensor)", x, y, z))
            G_frames.append(G_list)

    return G_frames, NG, Nframes

"""
Reads optivit and returns LED points for D and H frames along with counts.
"""
def read_optpivot(filepath):
    D_frames, H_frames = [], []
    with open(filepath, 'r') as f:
        parts = [p.strip() for p in f.readline().split(',')]
        ND, NH, Nframes = int(parts[0]), int(parts[1]), int(parts[2])

        for _ in range(Nframes):
            D_list, H_list = [], []
            for _i in range(ND):
                x, y, z = read_xyz(f.readline())
                D_list.append(Points("Optical D (base LEDs)", x, y, z))
            for _i in range(NH):
                x, y, z = read_xyz(f.readline())
                H_list.append(Points("Optical H (probe LEDs)", x, y, z))
            D_frames.append(D_list)
            H_frames.append(H_list)

    return D_frames, H_frames, ND, NH, Nframes

#----------------------------HW2----------------------------

"""
Reads CT fiducial files and returns b coordinates and count.
"""
def read_ctfiducials(filepath):
    b_coords = []
    with open(filepath, 'r') as f:
        # header = NB, filename
        parts = [p.strip() for p in f.readline().split(',')]
        NB = int(parts[0])

        for frame in range(NB):
            B_list = []
            for i in range(NG):
                x, y, z = read_xyz(f.readline())
                B_list.append(Points("CT Fiducials", x, y, z))
            b_coords.append(B_list)

    return b_coords, NB

"""
Reads EM fiducials and returns frames of contact in EM
"""
def read_emfiducials(filepath):
    B_frames = []
    with open(filepath, 'r') as f:
        # header = NG, Nframes, filename
        parts = [p.strip() for p in f.readline().split(',')]
        NB, Nframes = int(parts[0]), int(parts[1])

        for frame in range(Nframes):
            B_list = []
            for i in range(NB):
                x, y, z = read_xyz(f.readline())
                B_list.append(Points("EM Fiducials", x, y, z))
            B_frames.append(B_list)

    return B_frames, NB, Nframes

"""
Reads test points from EM.
"""
def read_emfiducials(filepath):
    B_frames = []
    with open(filepath, 'r') as f:
        # header = NG, Nframes, filename
        parts = [p.strip() for p in f.readline().split(',')]
        NB, Nframes = int(parts[0]), int(parts[1])

        for frame in range(Nframes):
            B_list = []
            for i in range(NB):
                x, y, z = read_xyz(f.readline())
                B_list.append(Points("EM Fiducials", x, y, z))
            B_frames.append(B_list)

    return B_frames, NB, Nframes


