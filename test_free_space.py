import numpy as np
import qpsolvers
from scipy import sparse
from free_space import FreeSpace

x = np.array([[1, 2], [3, 4]])

for i in x:
    print(i)

print(x - np.array([1, 2]))

# ----------------------------- QP -----------------------------
P = np.zeros((9, 9))
P[0:3, 0:3] = np.eye(3)
print(P)
P = sparse.csc_matrix(P)
print(P)
q = np.zeros(9)
print(q)

V = np.array([[1, -1, 0], [2, -1, 0], [1, 1, 0], [2, 1, 0], [2, -1, 1], [2, 1, 1]])
A = np.block([
    [np.diag(np.full(3, -1)),                  V.T],
    [np.zeros((1, 3)),          np.ones((1, 6))]
])
print(A)
A = sparse.csc_matrix(A)
print(A)
b = np.block([np.zeros(3), 1])
print(b)

G = np.block([np.zeros((6, 3)), np.diag(np.full(6, -1))])
print(G)
G = sparse.csc_matrix(G)
print(G)
h = np.zeros(6)
print(h)

x_opt = qpsolvers.solve_qp(P, q, G, h, A, b, solver="osqp")
print(x_opt)


# ----------------------------- clostest_point_on_obstacle and tangent_plane test -----------------------------
Cfree = FreeSpace(obstacles=[], pos0 = np.array([2, 0, 3]))
x_opt = Cfree.clostest_point_on_obstacle(V)
print("closest point = ", x_opt)
a, b = Cfree.tangent_plane(x_opt)
print("a = ", a, "\nb = ", b)

