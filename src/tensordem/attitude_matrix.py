"""Attitude matrix utilities for verification.

Provides quaternion-to-matrix conversion and orthogonality checks.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    
    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    
    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    w, x, y, z = q / np.linalg.norm(q)
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])
    return R


def rotation_matrix_orthogonality_error(R: np.ndarray) -> Tuple[float, float]:
    """Compute orthogonality error of a rotation matrix.
    
    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    
    Returns
    -------
    frobenius_err : float
        ||R^T R - I||_F (Frobenius norm).
    det_err : float
        |det(R) - 1|.
    """
    RTR = R.T @ R
    I3 = np.eye(3)
    frobenius_err = np.linalg.norm(RTR - I3, 'fro')
    det_err = abs(np.linalg.det(R) - 1.0)
    return frobenius_err, det_err


def attitude_matrix_from_quaternion_history(q_history: np.ndarray) -> np.ndarray:
    """Convert quaternion history to rotation matrix history.
    
    Parameters
    ----------
    q_history : ndarray, shape (N, 4)
        Quaternion history [w, x, y, z].
    
    Returns
    -------
    R_history : ndarray, shape (N, 3, 3)
        Rotation matrix history.
    """
    N = q_history.shape[0]
    R_history = np.zeros((N, 3, 3))
    for i in range(N):
        R_history[i] = quaternion_to_rotation_matrix(q_history[i])
    return R_history
