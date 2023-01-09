import numpy as np
from numpy import sin, cos
from scipy.spatial.transform import Rotation

def forward_kinematics_manipulator(q: np.ndarray) -> dict:
    """
    Compute the forward kinematics of the arm given the joints position.

    Args:
        q (np.ndarray): position of the joints of the manipulator

    Returns:
        dict: position and orientation of the end-point as a dictionary.
            `pos`: np.ndarray containing (x, y, z)
            `orientation`: np.ndarray containing the orientation in quaternions (qx, qy, qz, qw)
    """

    # Denavit-Hartenberg parameters
    d1 = 0.333
    d3 = 0.316
    d5 = 0.384
    d7 = 0.107
    a3 = 0.0825
    a4 = 0.0825
    a6 = 0.088
    q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

    r11 = cos(q7) * (sin(q6) * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) *\
        cos(q3)) - cos(q1) * cos(q4) * sin(q2)) - cos(q6) * (cos(q5) * (cos(q4) * (sin(q1) *\
        sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) + sin(q5) *\
        (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3)))) - sin(q7) * (sin(q5) * (cos(q4) *\
        (sin(q1) * sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) -\
        cos(q5) * (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3)))

    r12 = - sin(q7) * (sin(q6) * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) *\
        cos(q3)) - cos(q1) * cos(q4) * sin(q2)) - cos(q6) * (cos(q5) * (cos(q4) * (sin(q1) *\
        sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) + sin(q5) *\
        (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3)))) - cos(q7) * (sin(q5) * (cos(q4) *\
        (sin(q1) * sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) -\
        cos(q5) * (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3)))

    r13 = - cos(q6) * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) * cos(q3)) -\
        cos(q1) * cos(q4) * sin(q2)) - sin(q6) * (cos(q5) * (cos(q4) * (sin(q1) * sin(q3) -\
        cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) + sin(q5) * (cos(q3) *\
        sin(q1) + cos(q1) * cos(q2) * sin(q3)))

    r21 = sin(q7) * (sin(q5) * (cos(q4) * (cos(q1) * sin(q3) + cos(q2) * cos(q3) *\
        sin(q1)) - sin(q1) * sin(q2) * sin(q4)) - cos(q5) * (cos(q1) * cos(q3) - cos(q2) *\
        sin(q1) * sin(q3))) - cos(q7) * (sin(q6) * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) *\
        cos(q3) * sin(q1)) + cos(q4) * sin(q1) * sin(q2)) - cos(q6) * (cos(q5) * (cos(q4) *\
        (cos(q1) * sin(q3) + cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) +\
        sin(q5) * (cos(q1) * cos(q3) - cos(q2) * sin(q1) * sin(q3))))

    r22 = sin(q7) * (sin(q6) * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) * cos(q3) *\
        sin(q1)) + cos(q4) * sin(q1) * sin(q2)) - cos(q6) * (cos(q5) * (cos(q4) * (cos(q1) *\
        sin(q3) + cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) + sin(q5) *\
        (cos(q1) * cos(q3) - cos(q2) * sin(q1) * sin(q3)))) + cos(q7) * (sin(q5) * (cos(q4) *\
        (cos(q1) * sin(q3) + cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) -\
        cos(q5) * (cos(q1) * cos(q3) - cos(q2) * sin(q1) * sin(q3)))

    r23 = cos(q6) * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) * cos(q3) * sin(q1)) +\
        cos(q4) * sin(q1) * sin(q2)) + sin(q6) * (cos(q5) * (cos(q4) * (cos(q1) * sin(q3) +\
        cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) + sin(q5) * (cos(q1) *\
        cos(q3) - cos(q2) * sin(q1) * sin(q3)))

    r31 = sin(q7) * (sin(q5) * (cos(q2) * sin(q4) + cos(q3) * cos(q4) * sin(q2)) +\
        cos(q5) * sin(q2) * sin(q3)) + cos(q7) * (cos(q6) * (cos(q5) * (cos(q2) * sin(q4) +\
        cos(q3) * cos(q4) * sin(q2)) - sin(q2) * sin(q3) * sin(q5)) + sin(q6) * (cos(q2) *\
        cos(q4) - cos(q3) * sin(q2) * sin(q4)))

    r32 = cos(q7) * (sin(q5) * (cos(q2) * sin(q4) + cos(q3) * cos(q4) * sin(q2)) +\
        cos(q5) * sin(q2) * sin(q3)) - sin(q7) * (cos(q6) * (cos(q5) * (cos(q2) * sin(q4) +\
        cos(q3) * cos(q4) * sin(q2)) - sin(q2) * sin(q3) * sin(q5)) + sin(q6) * (cos(q2) *\
        cos(q4) - cos(q3) * sin(q2) * sin(q4)))

    r33 = sin(q6) * (cos(q5) * (cos(q2) * sin(q4) + cos(q3) * cos(q4) * sin(q2)) -\
        sin(q2) * sin(q3) * sin(q5)) - cos(q6) * (cos(q2) * cos(q4) - cos(q3) * sin(q2) * sin(q4))

    x = d5 * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) * cos(q3)) - cos(q1) *\
        cos(q4) * sin(q2)) - d7 * (cos(q6) * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) *\
        cos(q3)) - cos(q1) * cos(q4) * sin(q2)) + sin(q6) * (cos(q5) * (cos(q4) * (sin(q1) *\
        sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) + sin(q5) *\
        (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3)))) - d3 * cos(q1) * sin(q2) - a3 *\
        sin(q1) * sin(q3) + a6 * sin(q6) * (sin(q4) * (sin(q1) * sin(q3) - cos(q1) * cos(q2) *\
        cos(q3)) - cos(q1) * cos(q4) * sin(q2)) - a6 * cos(q6) * (cos(q5) * (cos(q4) * (sin(q1) *\
        sin(q3) - cos(q1) * cos(q2) * cos(q3)) + cos(q1) * sin(q2) * sin(q4)) + sin(q5) *\
        (cos(q3) * sin(q1) + cos(q1) * cos(q2) * sin(q3))) + a4 * cos(q4) * (sin(q1) * sin(q3) -\
        cos(q1) * cos(q2) * cos(q3)) + a3 * cos(q1) * cos(q2) * cos(q3) + a4 * cos(q1) * sin(q2) * sin(q4)

    y = d7 * (cos(q6) * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) * cos(q3) * sin(q1)) +\
        cos(q4) * sin(q1) * sin(q2)) + sin(q6) * (cos(q5) * (cos(q4) * (cos(q1) * sin(q3) +\
        cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) + sin(q5) * (cos(q1) *\
        cos(q3) - cos(q2) * sin(q1) * sin(q3)))) - d5 * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) *\
        cos(q3) * sin(q1)) + cos(q4) * sin(q1) * sin(q2)) + a3 * cos(q1) * sin(q3) - d3 * sin(q1) *\
        sin(q2) - a6 * sin(q6) * (sin(q4) * (cos(q1) * sin(q3) + cos(q2) * cos(q3) * sin(q1)) +\
        cos(q4) * sin(q1) * sin(q2)) + a6 * cos(q6) * (cos(q5) * (cos(q4) * (cos(q1) * sin(q3) +\
        cos(q2) * cos(q3) * sin(q1)) - sin(q1) * sin(q2) * sin(q4)) + sin(q5) * (cos(q1) *\
        cos(q3) - cos(q2) * sin(q1) * sin(q3))) - a4 * cos(q4) * (cos(q1) * sin(q3) + cos(q2) *\
        cos(q3) * sin(q1)) + a3 * cos(q2) * cos(q3) * sin(q1) + a4 * sin(q1) * sin(q2) * sin(q4)

    z = d1 + d7 * (sin(q6) * (cos(q5) * (cos(q2) * sin(q4) + cos(q3) * cos(q4) *\
        sin(q2)) - sin(q2) * sin(q3) * sin(q5)) - cos(q6) * (cos(q2) * cos(q4) - cos(q3) *\
        sin(q2) * sin(q4))) + d5 * (cos(q2) * cos(q4) - cos(q3) * sin(q2) * sin(q4)) + d3 *\
        cos(q2) + a6 * sin(q6) * (cos(q2) * cos(q4) - cos(q3) * sin(q2) * sin(q4)) + a3 * cos(q3) *\
        sin(q2) - a4 * cos(q2) * sin(q4) + a6 * cos(q6) * (cos(q5) * (cos(q2) * sin(q4) + cos(q3) *\
        cos(q4) * sin(q2)) - sin(q2) * sin(q3) * sin(q5)) - a4 * cos(q3) * cos(q4) * sin(q2)

    rotational_matrix = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    r = Rotation.from_matrix(rotational_matrix)
    r = r.as_quat()

    pose = {'pos': np.array([x, y, z]), 'orientation': r}

    return pose