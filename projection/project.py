from typing import *
import math
import numpy as np

def project3d(p1: np.ndarray, p2: np.ndarray, dir: str, world_scaling:int=1):
    stereo_calibration = np.load("{}/extrinsics/stereo_calibration.npz".format(dir))
    intrinsics_1 = np.load("{}/intrinsics/camera_calibration_0.npz".format(dir))
    intrinsics_2 = np.load("{}/intrinsics/camera_calibration_1.npz".format(dir))
    K1 = intrinsics_1["calibration_mtx"]
    K2 = intrinsics_2["calibration_mtx"]
    R, T = stereo_calibration['R'], stereo_calibration['T']
    
    # projects from camera 2 to camera 1
    extrinsic = np.hstack([R, T])
    projection_matrix_1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
    projection_matrix_2 = np.dot(K2, extrinsic)
    
    pcd = [triangulate(projection_matrix_1, projection_matrix_2, p1[i, :], p2[i, :]) for i in range(p1.shape[0])]
    numpy_pcd = np.array(pcd)
    return numpy_pcd


def triangulate(P1, P2, point1, point2):
    point1 = point1.reshape(-1)
    point2 = point2.reshape(-1)
    A = np.array([point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ])
    A = A.reshape((4,4))
    
    B = A.T @ A
    U, s, Vh = np.linalg.svd(B, full_matrices = False)

    point_3d = Vh[3,0:3]/Vh[3,3]
    return point_3d

