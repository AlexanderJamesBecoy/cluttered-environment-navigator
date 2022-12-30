import numpy as np
from casadi import *
import qpsolvers
from scipy import sparse


EPSILON_SPHERE = 0.1
# SPACE_DIM = 2
SPACE_DIM = 3
MAX_ITER = 10
TOLLERANCE = 0.02


class Ellipsoid:
    """
    Class that implement an ellipsoid
    """

    def __init__(self, center: np.ndarray = np.zeros(SPACE_DIM), matrix_C: np.ndarray = np.eye(SPACE_DIM)) -> None:
        """
        Initialize the ellipsoid in two different forms:
        ellipsoid = {x = C*y + d | ||d|| <= 1}
        ellipsoid = {x | (x - d)^T * C^-1 * C^-T * (x - d)}

        Args:
            center (np.ndarray, optional): center of the ellipsoid. Defaults to np.zeros(SPACE_DIM).
            matrix_C (np.ndarray, optional): matrix of the ellipsoid axes. Defaults to np.eye(SPACE_DIM).
        """

        self.d = center
        self.C = matrix_C
        self.C_inv = np.linalg.inv(self.C)

        pass


class FreeSpace:
    """
    This class implement an algorithm to find a large obstacle-free convex region.
    The code is based on the paper:
    Deits and Tedrake 2015, Computing Large Convex Regions of Obstacle-Free Space through Semidefinite Programming
    """

    def __init__(self, obstacles: list, pos0: np.ndarray = np.zeros(SPACE_DIM)) -> None:
        """
        Initialize the free space, described by the hyperplanes {x | A*x <= b} and the ellipsoid.

        Args:
            obstacles (list): list of obstacles, each element of the list contains the vertices on the related obstacle
            pos0 (np.ndarray, optional): initial position of the center of the ellipsoid. Defaults to np.zeros(SPACE_DIM).
        """

        self.ellipsoid = Ellipsoid(pos0, np.eye(SPACE_DIM)*EPSILON_SPHERE)
        self.ostacles = obstacles
        self.A = []
        self.b = []

        pass

    def update_free_space(self, pos0: np.ndarray = np.zeros(SPACE_DIM)):

        # re-initialize the ellipsoid to a ball and the hyperplanes
        self.ellipsoid = Ellipsoid(pos0, np.eye(SPACE_DIM)*EPSILON_SPHERE)
        self.A = []
        self.b = []

        # keep iterating the algorithm
        for _ in range(MAX_ITER):

            # keep track of the previous determinant to check the tolerance on the relative change in ellipsoid volume
            det_C_prec = np.linalg.det(self.ellipsoid.C)
            self.separating_hyperplanes() # find hyperplanes that separates the obstacles from the ellipsoid
            self.inscribed_ellipsoid() # find the maximum volume ellipsoid inscribed in the hyperplanes
            det_C = np.linalg.det(self.ellipsoid.C)
            if ((det_C - det_C_prec) / det_C_prec) < TOLLERANCE: # check termination condition
                break

        pass 

    def separating_hyperplanes(self):

        n_obs = len(self.obstacles)
        obs_excluded = []
        obs_remaining = list(range(0, n_obs))

        while len(obs_remaining) != 0:

            # find the closest obstacles to the ellipsoid
            closest_obs = self.clostest_obstacle(self, obs_remaining)
            # find the closest point of the obstacle to the ellipsoid
            x_closest = self.clostest_point_on_obstacle(self, closest_obs)
            # find the hyperplane tangent the point that separates the obstacle from the ellipsoid
            a_i, b_i = self.tangent_plane(self, x_closest)
            self.A.append(a_i)
            self.b.append(b_i)

            # check if the hyperplane found separes also other obstacles from the ellipsoid
            for obs_i in obs_remaining:
                check = True
                for vertex_j in self.ostacles[obs_i]:
                    if a_i @ vertex_j < b_i:
                        check = False

                if check:
                    obs_remaining.remove(obs_i)
                    obs_excluded.append(obs_i)

        pass

    def inscribed_ellipsoid(self):

        pass

    def clostest_obstacle(self, obs_remaining):

        pass

    def clostest_point_on_obstacle(self, obstacle: np.ndarray) -> np.ndarray:

        num_vertices = obstacle.shape[0] # number of vertices in the obstacle
        vertices_j = self.ellipsoid.C_inv @ (obstacle - self.ellipsoid.d).T  # transformed vertices in ball space

        # Quadratic Programming formulation:
        # min           0.5 x*P*x + q*x
        # subject to    G*x <= h, A*x = b, lb <= x <= ub

        num_var = SPACE_DIM + num_vertices # number of optimization variables
        P = P = np.zeros((num_var, num_var))
        P[0:SPACE_DIM, 0:SPACE_DIM] = np.eye(SPACE_DIM)
        P = sparse.csc_matrix(P) # for best performance, build the matrix as a sparse matrix
        q = np.zeros(num_var)
        G = np.block([np.zeros((num_vertices, 3)), np.diag(np.full(num_vertices, -1))])
        G = sparse.csc_matrix(G)
        h = np.zeros(num_vertices)
        A = np.block([
            [np.diag(np.full(SPACE_DIM, -1)),                   vertices_j],
            [np.zeros((1, SPACE_DIM)),          np.ones((1, num_vertices))]
        ])
        A = sparse.csc_matrix(A)
        b = np.block([np.zeros(SPACE_DIM), 1])

        x_opt = qpsolvers.solve_qp(P, q, G, h, A, b, solver="osqp") # solve the problem
        x_opt = x_opt[0:SPACE_DIM] # select only the position of the closest point among the optimization variables
        x_closest = self.ellipsoid.C @ x_opt + self.ellipsoid.d

        return x_closest

    def tangent_plane(self, x) -> tuple:

        a_i = 2 * self.ellipsoid.C_inv @ self.ellipsoid.C_inv.T @ (x - self.ellipsoid.d).T
        b_i = x @ a_i

        return a_i, b_i
