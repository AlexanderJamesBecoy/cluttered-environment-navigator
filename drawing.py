import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits import mplot3d
import numpy as np
from free_space import FreeSpace
import time
from free_space import Ellipsoid


def draw_region(obstacles: list[np.ndarray], ellipsoid: Ellipsoid, pos0: list, n_points: int=50):

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
    points = random_points_on_sphere(radius, n_points)

    for point in points:
        ell_points = ellipsoid.C @ np.array(point) + ellipsoid.d
        # print(ell_points)
        ax.scatter(ell_points[0], ell_points[1], ell_points[2], color='blue')

    ax.scatter(pos0[0], pos0[1], pos0[2], color='orange')

    plt.show()