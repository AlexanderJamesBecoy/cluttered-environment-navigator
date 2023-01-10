import numpy as np
import qpsolvers
from scipy import sparse
from free_space import FreeSpace
import cvxpy as cp
import time

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

ob1 = np.array([[-10, -10, 1], [-10, 10 , 1], [10, -10, 1], [10, 10, 1], 
[-10, -10, 1.1], [-10, 10 , 1.1], [10, -10, 1.1], [10, 10, 1.1]]) # ceiling

ob2 = np.array([[-10, -10, 0], [-10, 10 , 0], [10, -10, 0], [10, 10, 0],
[-10, -10, -0.1], [-10, 10 , -0.1], [10, -10, -0.1], [10, 10, -0.1]]) # floor

ob3 = np.array([[-10, -10, 1], [-10, 10 , 1], [-10, -10, 0], [-10, 10 , 0],
[-10.1, -10, 1], [-10.1, 10 , 1], [-10.1, -10, 0], [-10.1, 10 , 0]])# wall 1

ob4 = np.array([[-10, -10, 1], [10, -10, 1], [-10, -10, 0], [10, -10, 0],
[-10, -10.1, 1], [10, -10.1, 1], [-10, -10.1, 0], [10, -10.1, 0]]) # wall 2

ob5 = np.array([[10, -10, 1], [10, 10, 1], [10, -10, 0], [10, 10, 0],
[10.1, -10, 1], [10.1, 10, 1], [10.1, -10, 0], [10.1, 10, 0]]) # wall 3

ob6 = np.array([[-10, 10, 1], [10, 10, 1], [-10, 10, 0], [10, 10, 0],
[-10, 10.1, 1], [10, 10.1, 1], [-10, 10.1, 0], [10, 10.1, 0]]) # wall 4


ob7 = np.array([[3, 3, 0], [0, 3, 0], [3, 5, 0], [3, 3, 0.5]])
ob8 = np.array([[-5, -5, 0], [0, -5, 0], [0, -2, 0], [-5, -2, 0], [-5, -5, 0.8], [0, -5, 0.8], [0, -2, 0.8], [-5, -2, 0.8]])

obstacles = [ob1, ob2, ob3, ob4, ob5, ob6, ob7, ob8]

# Cfree = FreeSpace(obstacles)

# start_time = time.time()
# A, b = Cfree.update_free_space(np.array([0, 0, 0.5]))
# end_time = time.time()
# print(A)
# print(b)
# print("Time required: ", (end_time - start_time))


# ----------------------------- visualization -----------------------------

import matplotlib.pyplot as plt

def draw_line(x, y):
  # plot the points using matplotlib
  plt.plot(x, y)
 
  # show the plot
  plt.show()

# test the function
x = [0, 1, 2, 3]
y = [0, 1, 4, 9]
draw_line(x, y)



