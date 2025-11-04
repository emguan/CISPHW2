"""
Finds rigid frame transformation (F) between 2 point clouds/Points sets.

Author: Emily Guan
"""

from utils.mathpackage.mathpackage import Points, Transformations, Rotations
import numpy as np

"""
Return the centroid as a Points in the common frame.
"""
def find_centroids(pts: list[Points]) -> Points:
    
    frames = {p.frame for p in pts if p.frame}
    frame = frames.pop() if len(frames) == 1 else ""

    # centroid = 1/N \sum points
    arr = np.stack([p.points_3d() for p in pts], axis=0) 
    c = arr.mean(axis=0)
    return Points.from_array(frame, c)


"""
Compute the rigid transformation (R, p) that aligns point set A to B
such that:  b_i ≈ R * a_i + p

Using Arun's method: https://jingnanshi.com/blog/arun_method_for_3d_reg.html 
"""
def aruns_method(a: list[Points], b: list[Points]) -> Transformations:

    # initialize A and B for MA = B
    A = np.stack([p.points_3d() for p in a], axis=0)
    B = np.stack([p.points_3d() for p in b], axis=0)

    # find centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # center the point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # compute H
    H = A_centered.T @ B_centered

    # orthogonalize to get R0 & "iterate" via svd
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # handle reflection (verify is rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # compute translation
    t = centroid_B - R @ centroid_A

    # create frame name
    frames = {p.frame for p in a + b if p.frame}
    frame = frames.pop() if len(frames) == 1 else ""

    return Transformations(frame, Rotations(R), Points.from_array(frame, t))