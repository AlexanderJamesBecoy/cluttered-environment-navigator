import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits import mplot3d
import numpy as np
import qpsolvers
from scipy import sparse
from controller.free_space import FreeSpace
import cvxpy as cp
import time

print(cp.installed_solvers())

# ----------------------------- various tests -----------------------------

# print("Solvers cvxpy: ", cp.installed_solvers())


# x = np.array([[1, 2], [3, 4], [5, 6]])

# for i in x[1:]:
#     print(i)

# print(x - np.array([1, 2]))


# y = np.array([1, 2, 3])
# for a, b in zip(x, y):
#     print(a, b)


# ----------------------------- QP -----------------------------

# P = np.zeros((9, 9))
# P[0:3, 0:3] = np.eye(3)
# print(P)
# P = sparse.csc_matrix(P)
# print(P)
# q = np.zeros(9)
# print(q)

# V = np.array([[1, -1, 0], [2, -1, 0], [1, 1, 0], [2, 1, 0], [2, -1, 1], [2, 1, 1]])
# A = np.block([
#     [np.diag(np.full(3, -1)),                  V.T],
#     [np.zeros((1, 3)),          np.ones((1, 6))]
# ])
# print(A)
# A = sparse.csc_matrix(A)
# print(A)
# b = np.block([np.zeros(3), 1])
# print(b)

# G = np.block([np.zeros((6, 3)), np.diag(np.full(6, -1))])
# print(G)
# G = sparse.csc_matrix(G)
# print(G)
# h = np.zeros(6)
# print(h)

# x_opt = qpsolvers.solve_qp(P, q, G, h, A, b, solver="osqp")
# print(x_opt)


# ----------------------------- clostest_point_on_obstacle and tangent_plane test -----------------------------

# V = np.array([[1, -1, 0], [2, -1, 0], [1, 1, 0], [2, 1, 0], [2, -1, 1], [2, 1, 1]])
# Cfree = FreeSpace(obstacles=[], pos0 = np.array([2, 0, 3]))
# x_opt = Cfree.clostest_point_on_obstacle(V)
# print("closest point = ", x_opt)
# a, b = Cfree.tangent_plane(x_opt)
# print("a = ", a, "\nb = ", b)

# ----------------------------- full test 1 -----------------------------

print("\n\n\n\n\n")

ob1 = np.array([[-10, -10, 1], [-10, 10, 1], [10, -10, 1], [10, 10, 1],
                [-10, -10, 1.1], [-10, 10, 1.1], [10, -10, 1.1], [10, 10, 1.1]])  # ceiling

ob2 = np.array([[-10, -10, 0], [-10, 10, 0], [10, -10, 0], [10, 10, 0],
                [-10, -10, -0.1], [-10, 10, -0.1], [10, -10, -0.1], [10, 10, -0.1]])  # floor

ob3 = np.array([[-10, -10, 1], [-10, 10, 1], [-10, -10, 0], [-10, 10, 0],
                [-10.1, -10, 1], [-10.1, 10, 1], [-10.1, -10, 0], [-10.1, 10, 0]])  # wall 1

ob4 = np.array([[-10, -10, 1], [10, -10, 1], [-10, -10, 0], [10, -10, 0],
                [-10, -10.1, 1], [10, -10.1, 1], [-10, -10.1, 0], [10, -10.1, 0]])  # wall 2

ob5 = np.array([[10, -10, 1], [10, 10, 1], [10, -10, 0], [10, 10, 0],
                [10.1, -10, 1], [10.1, 10, 1], [10.1, -10, 0], [10.1, 10, 0]])  # wall 3

ob6 = np.array([[-10, 10, 1], [10, 10, 1], [-10, 10, 0], [10, 10, 0],
                [-10, 10.1, 1], [10, 10.1, 1], [-10, 10.1, 0], [10, 10.1, 0]])  # wall 4


ob7 = np.array([[3, 3, 0], [0, 3, 0], [3, 5, 0], [0, 5, 0], [
               3, 3, 0.5], [0, 3, 0.5], [3, 5, 0.5], [0, 5, 0.5]])
ob8 = np.array([[-5, -5, 0], [0, -5, 0], [0, -2, 0], [-5, -2, 0],
               [-5, -5, 0.8], [0, -5, 0.8], [0, -2, 0.8], [-5, -2, 0.8]])

obstacles = [ob1, ob2, ob3, ob4, ob5, ob6, ob7, ob8]

floor = np.array([[4, 4, 0], [-4, 4, 0], [-4, -4, 0], [4, -4, 0],
                 [4, 4, -0.1], [-4, 4, -0.1], [-4, -4, -0.1], [4, -4, -0.1]])
ceiling = np.array([[4, 4, 1.5], [-4, 4, 1.5], [-4, -4, 1.5], [4, -4, 1.5],
                   [4, 4, 1.6], [-4, 4, 1.6], [-4, -4, 1.6], [4, -4, 1.6]])
Hlow = np.array([[-3.75, -3.7, 0], [3.75, -3.7, 0], [3.75, -3.8, 0], [-3.75, -3.8, 0],
                 [-3.75, -3.7, 1.5], [3.75, -3.7, 1.5], [3.75, -3.8, 1.5], [-3.75, -3.8, 1.5]])
Hhigh = np.array([[-3.75, 3.7, 0], [3.75, 3.7, 0], [3.75, 3.8, 0], [-3.75, 3.8, 0],
                  [-3.75, 3.7, 1.5], [3.75, 3.7, 1.5], [3.75, 3.8, 1.5], [-3.75, 3.8, 1.5]])
Vleft = np.array([[-3.8, 3.75, 0], [-3.7, 3.75, 0], [-3.7, -3.75, 0], [-3.8, -3.75, 0],
                  [-3.8, 3.75, 1.5], [-3.7, 3.75, 1.5], [-3.7, -3.75, 1.5], [-3.8, -3.75, 1.5]])
Vright = np.array([[3.8, 3.75, 0], [3.7, 3.75, 0], [3.7, -3.75, 0], [3.8, -3.75, 0],
                   [3.8, 3.75, 1.5], [3.7, 3.75, 1.5], [3.7, -3.75, 1.5], [3.8, -3.75, 1.5]])
block = np.array([[0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0],
                  [0.5, -0.5, 1.5], [0.5, 0.5, 1.5], [-0.5, 0.5, 1.5], [-0.5, -0.5, 1.5]])
obstacles = [floor, ceiling, Hlow, Hhigh, Vleft, Vright, block]
p0 = [-2.2915978518991427, -2.021272858731986, 0.4]

Cfree = FreeSpace(obstacles)

start_time = time.time()
A, b = Cfree.update_free_space(p0) # np.array([0, 0, 0.5])
end_time = time.time()
print(A)
print(b)
print("Time required: ", (end_time - start_time))


# ----------------------------- visualization -----------------------------

fig = plt.figure()
ax = plt.axes(projection='3d')

for obj in obstacles:

    hull = ConvexHull(obj)

    for simplex in hull.simplices:
        ax.plot(obj[simplex, 0], obj[simplex, 1], obj[simplex, 2], 'r-')

    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2])


def random_points_on_sphere(r, n_points):
    points = []
    for _ in range(n_points):
        azimuth = random.uniform(0, 2 * np.pi)
        polar = random.uniform(0, np.pi)
        x = r * np.sin(polar) * np.cos(azimuth)
        y = r * np.sin(polar) * np.sin(azimuth)
        z = r * np.cos(polar)
        points.append((x, y, z))
    return np.array(points)


radius = 1
n_points = 200
points = random_points_on_sphere(radius, n_points)

for point in points:
    ell_points = Cfree.ellipsoid.C @ np.array(point) + Cfree.ellipsoid.d
    # print(ell_points)
    ax.scatter(ell_points[0], ell_points[1], ell_points[2], color='blue')

ax.scatter(p0[0], p0[1], p0[2], color='blue')

plt.show()
