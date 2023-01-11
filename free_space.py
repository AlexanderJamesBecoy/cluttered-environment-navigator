import numpy as np
from casadi import *
import qpsolvers
from scipy import sparse
import cvxpy as cp
import time


EPSILON_SPHERE = 0.1
# SPACE_DIM = 2
SPACE_DIM = 3
MAX_ITER = 5
TOLLERANCE = 0.02
CHECK_TOLLERANCE = 0.01


class Ellipsoid:
    """
    Class that implement an ellipsoid
    """

    def __init__(self, center: np.ndarray = np.zeros(SPACE_DIM), matrix_C: np.ndarray = np.eye(SPACE_DIM)) -> None:
        """
        Initialize the ellipsoid in two different forms:
        ellipsoid = {x = C*y + d | ||y|| <= 1}
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
        self.obstacles = obstacles
        self.A = []
        self.b = []

        pass

    def update_free_space(self, pos0) -> tuple:

        # re-initialize the ellipsoid to a ball and the hyperplanes
        self.ellipsoid = Ellipsoid(pos0, np.eye(SPACE_DIM)*EPSILON_SPHERE)
        self.A = []
        self.b = []

        # keep iterating the algorithm
        for i in range(MAX_ITER):

            print("Iteration number: ", i+1, "/", MAX_ITER)
            # keep track of the previous determinant to check the tolerance on the relative change in ellipsoid volume
            det_C_prec = np.linalg.det(self.ellipsoid.C)
            
            print("Computing separating hyperplanes...")
            start_time = time.time()
            self.separating_hyperplanes() # find hyperplanes that separates the obstacles from the ellipsoid
            end_time = time.time()
            print("Time: ", (end_time - start_time))

            print("Computing inscribed ellipsoid...")
            start_time = time.time()
            self.inscribed_ellipsoid() # find the maximum volume ellipsoid inscribed in the hyperplanes
            end_time = time.time()
            print("Time: ", (end_time - start_time))

            det_C = np.linalg.det(self.ellipsoid.C)

            relative_increase = (det_C - det_C_prec) / det_C_prec
            print("Relative increase: ", relative_increase)
            if relative_increase < TOLLERANCE: # check termination condition
                print("Update succeeded!")
                break

        return self.A, self.b 

    def separating_hyperplanes(self):

        n_obs = len(self.obstacles)
        obs_excluded = []
        obs_remaining = list(range(0, n_obs))
        self.A = []
        self.b = []

        while len(obs_remaining) != 0:

            # find the closest obstacles to the ellipsoid
            # print("Looking for closest obstacle...")
            closest_obs = self.clostest_obstacle(obs_remaining)
            # find the closest point of the obstacle to the ellipsoid
            # print("Looking for closest point...")
            x_closest = self.clostest_point_on_obstacle(closest_obs)
            # find the hyperplane tangent to the point that separates the obstacle from the ellipsoid
            # print("Computing separating hyperplane...")
            a_i, b_i = self.tangent_plane(x_closest)
            self.A.append(a_i)
            self.b.append(b_i)

            # check if the hyperplane found separes also other obstacles from the ellipsoid
            for obs_i in obs_remaining:
                check = True
                for vertex_j in self.obstacles[obs_i]:
                    if a_i @ vertex_j < (b_i - CHECK_TOLLERANCE):
                        check = False
                        break

                if check:
                    # print("Obstacle removed...")
                    obs_remaining.remove(obs_i)
                    obs_excluded.append(obs_i)

        pass

    def inscribed_ellipsoid(self):

        # Largest volume inner ellipsoid problem formulation:
        # max               log det(C)
        # subject to        ||C*ai|| + ai^T * d <= bi for all i
        #                   C >> 0
        # print("Computing inscribed ellipsoid...")
        C = cp.Variable((SPACE_DIM, SPACE_DIM), symmetric=True)
        d = cp.Variable(SPACE_DIM)
        objective = cp.Maximize(cp.log_det(C))
        constraints = [C >> 0]
        for ai, bi in zip(self.A, self.b):
            constraints += [cp.norm(C @ ai) + ai @ d <= bi]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        # print("Solution: ", C.value, d.value)

        # Update the ellipsoid parameters
        self.ellipsoid = Ellipsoid(d.value, C.value)

        pass

    def clostest_obstacle(self, obs_remaining) -> np.ndarray:

        index_closest = obs_remaining[0]
        min_dist = 100 # initialization of the maximum distance
        for obs in obs_remaining:
            # initialize the closest distance of the obstacle by considering the first vertex
            dist_obs = self.calculate_min_dist(self.obstacles[obs][0])
            for vertex_j in self.obstacles[obs][1:]:
                dist = self.calculate_min_dist(vertex_j)
                if dist < dist_obs:
                    dist_obs = dist
            
            if dist_obs < min_dist:
                min_dist = dist_obs
                index_closest = obs
        
        return self.obstacles[index_closest]

    def calculate_min_dist(self, point: np.ndarray) -> float:

        # TODO: for now I considered only the distance from the center of the ellipsoid
        min_dist = np.linalg.norm(self.ellipsoid.d - point)

        return min_dist

    def clostest_point_on_obstacle(self, obstacle: np.ndarray) -> np.ndarray:

        num_vertices = obstacle.shape[0] # number of vertices in the obstacle
        vertices_j = self.ellipsoid.C_inv @ (obstacle - self.ellipsoid.d).T  # transformed vertices in ball space

        # Quadratic Programming formulation:
        # min           0.5 x*P*x + q*x
        # subject to    G*x <= h, A*x = b, lb <= x <= ub
        # https://pypi.org/project/qpsolvers/

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
        x_closest = self.ellipsoid.C @ x_opt + self.ellipsoid.d # apply inverse transformation to ellipsoide space

        return x_closest

    def tangent_plane(self, x) -> tuple:

        a_i = 2 * self.ellipsoid.C_inv @ self.ellipsoid.C_inv.T @ (x - self.ellipsoid.d).T
        b_i = x @ a_i

        return a_i, b_i
