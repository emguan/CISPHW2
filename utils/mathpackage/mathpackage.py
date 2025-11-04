"""
Contains math package that covers all 3D points, rotations, and frame transformations.

Author: Emily Guan, Brian Sun
"""

import numpy as np

"""

Representation for 3D points. Contains frame & coordinates relative to frame.

"""
class Points:

    def __init__(self, frame: str = "", x=0.0, y=0.0, z=0.0):
        self.frame = frame
        self.x = float(x); self.y = float(y); self.z = float(z)

    '''
    Converts array to Points object.
    '''
    def from_array(frame: str, arr: np.ndarray):
        arr = np.asarray(arr, float).reshape(3)
        return Points(frame, arr[0], arr[1], arr[2])

    '''
    Returns 3d point representation of Points in ndarray.
    '''
    def points_3d(self):
        return np.array([self.x, self.y, self.z], dtype=float)
    
    '''
    Addition of 2 Points.
    '''
    def __add__(self, other: "Points"):
        return Points.from_array(self.frame or other.frame, self.points_3d() + other.points_3d())

    '''
    Subtraction of 2 Points.
    '''
    def __sub__(self, other: "Points"):
        return Points.from_array(self.frame or other.frame, self.points_3d() - other.points_3d())

    '''
    Multiplication of 2 Points.
    '''
    def __mul__(self, k: float):
        return Points.from_array(self.frame, self.points_3d() * float(k))

    "Division of 2 Points"
    def __truediv__(self, k: float):
        return Points.from_array(self.frame, self.points_3d() / float(k))


    __rmul__ = __mul__


"""

Representation for Rotations. Contains 3x3 rotation matrix.

"""
class Rotations:

    def __init__(self, R: np.ndarray):
        if R.shape != (3,3):
            raise ValueError(f"R must be 3x3")
        self.R = R

    '''
    Builds Rotation object from 3 angles.
    '''
    def build(alpha: float, beta: float, gamma: float) -> "Rotations":
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta),  np.sin(beta)
        cc, sc = np.cos(gamma), np.sin(gamma)
        Rx = np.array([[1, 0, 0],
                       [0, ca, -sa],
                       [0, sa,  ca]])
        Ry = np.array([[ cb, 0, sb],
                       [  0, 1,  0],
                       [-sb, 0, cb]])
        Rz = np.array([[ cc, -sc, 0],
                       [ sc,  cc, 0],
                       [  0,   0, 1]])
        return Rotations(R=Rz @ Ry @ Rx)

    '''
    Take inverse of rotation (by taking inverse of rotation matrix, which is also the transpose)
    '''
    def inverse(self) -> "Rotations":
        return Rotations(self.R.T)

"""

Frame Transformation Class. 

"""
class Transformations:
    def __init__(self, frame: str, r: Rotations, p: Points):
        self.frame = frame
        self.r = r
        self.p = p 

    '''
    Take inverse of frame transformation.
    '''
    def inverse(self) -> "Transformations":
        R_inv = self.r.R.T # R = R^{-1}
        t_inv = -R_inv @ self.p.points_3d() #p = R^{-1} • p
        return Transformations(self.frame, Rotations(R_inv), Points(self.frame, *t_inv))

    '''
    Compose transforms: self • other (apply other first, then self).
    ''' 
    def __matmul__(self, other: "Transformations") -> "Transformations":
        
        R_new = self.r.R @ other.r.R
        t_new = self.r.R @ other.p.points_3d() + self.p.points_3d()
        return Transformations(self.frame, Rotations(R_new), Points(self.frame, *t_new))

    """
    Apply transform to 3d vector of point.
    """
    def apply(self, P: np.ndarray) -> np.ndarray:
        t = self.p.points_3d()
        if P.ndim == 1:
            return self.r.R @ P + t
        elif P.ndim == 2 and P.shape[1] == 3:
            return (P @ self.r.R.T) + t
        else:
            raise ValueError("P must be shape (3,) or (N,3)")


