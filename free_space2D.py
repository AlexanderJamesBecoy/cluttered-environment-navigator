import numpy as np
from free_space import FreeSpace
from shapely import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random

CHECK_TOLLERANCE = 0.05

class FreeSpace2D(FreeSpace):
    def __init__(self, obstacles: list, pos0: np.ndarray) -> None:
        super().__init__(obstacles, pos0)
        self.x_closest_cluster = []
        self.union = None
        self.union_ellipse = []

    def separating_hyperplanes(self):

        n_obs = len(self.obstacles)
        obs_excluded = []
        obs_remaining = list(range(0, n_obs))
        self.A = []
        self.b = []
        self.x_closest = []

        while len(obs_remaining) != 0:

            # find the closest obstacles to the ellipsoid
            # print("Looking for closest obstacle...")
            index_closest, closest_obs = self.closest_obstacle(obs_remaining)
            # find the closest point of the obstacle to the ellipsoid
            # print("Looking for closest point...")
            x_closest = self.closest_point_on_obstacle(closest_obs)
            self.x_closest.append(x_closest)
            # find the hyperplane tangent to the point that separates the obstacle from the ellipsoid
            # print("Computing separating hyperplane...")
            a_i, b_i = self.tangent_plane(x_closest)
            self.A.append(a_i)
            self.b.append(b_i)

            # the closest obstacle can be removed from the set (since we have found its hyperplane)
            obs_remaining.remove(index_closest)
            obs_excluded.append(index_closest)


            # check if the hyperplane found separes also other obstacles from the ellipsoid
            for obs_i in obs_remaining:
                check = True
                for vertex_j in self.obstacles[obs_i]:
                    if a_i @ vertex_j < (b_i - CHECK_TOLLERANCE):
                        check = False
                        break

                if check and obs_i != index_closest:
                    # print("Obstacle removed...")
                    obs_remaining.remove(obs_i)
                    obs_excluded.append(obs_i)

    def combine_ellips_2D(self, vertices, current_ellipsoid: np.ndarray, next_ellipsoid: np.ndarray):
        """
            Combines 2 ellipsoids. Takes 2 arrays of points closest to that ellipsoid as input.
        """
        # Collapse to 2D points
        plt.figure()
        ax = plt.axes()
        ax = plt.axes(projection='3d')

        for obj in vertices:
            hull = ConvexHull(obj)

            for simplex in hull.simplices:
                ax.plot(obj[simplex, 0], obj[simplex, 1], obj[simplex, 2], 'g-')

            # ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2])
        self.union_ellipse = np.concatenate((current_ellipsoid, next_ellipsoid), axis=0)

        hull = ConvexHull(self.union_ellipse)
        for simplex in hull.simplices:
            ax.plot(self.union_ellipse[simplex, 0], self.union_ellipse[simplex, 1], self.union_ellipse[simplex, 2], 'r-')

        # hull = ConvexHull(current_ellipsoid)
        # for simplex in hull.simplices:
        #     ax.plot(current_ellipsoid[simplex, 0], current_ellipsoid[simplex, 1], current_ellipsoid[simplex, 2], 'b-')

        # hull = ConvexHull(next_ellipsoid)
        # for simplex in hull.simplices:
        #     ax.plot(next_ellipsoid[simplex, 0], next_ellipsoid[simplex, 1], next_ellipsoid[simplex, 2], 'r-')
        
        # hull = ConvexHull(current_ellipsoid[:, :2])
        # for simplex in hull.simplices:
        #     ax.plot(current_ellipsoid[simplex, 0], current_ellipsoid[simplex, 1], 'b-')

        # hull = ConvexHull(next_ellipsoid[:, :2])
        # for simplex in hull.simplices:
        #     ax.plot(next_ellipsoid[simplex, 0], next_ellipsoid[simplex, 1], 'r-')

        # def random_points_on_sphere(r, n_points):
        #     points = []
        #     for _ in range(n_points):
        #         azimuth = random.uniform(0, 2 * np.pi)
        #         polar = random.uniform(0, np.pi)
        #         x = r * np.sin(polar) * np.cos(azimuth)
        #         y = r * np.sin(polar) * np.sin(azimuth)
        #         z = r * np.cos(polar)
        #         points.append((x, y, z))
        #     return np.array(points)


        # radius = 1
        # n_points = 50
        # points = random_points_on_sphere(radius, n_points)

        # for point in points:
        #     ell_points = self.ellipsoid.C @ np.array(point) + self.ellipsoid.d
        #     # print(ell_points)
        #     ax.scatter(ell_points[0], ell_points[1], ell_points[2], color='blue')

        # current_ellipsoid = np.delete(current_ellipsoid, np.argmax(current_ellipsoid[:, 2]), axis = 0)
        # current_ellipsoid = np.delete(current_ellipsoid, np.argmin(current_ellipsoid[:, 2]), axis = 0)
        # Poly1 = Polygon(current_ellipsoid)
        # x, y = Poly1.exterior.xy
        # ax.plot(x, y)
        # next_ellipsoid = np.delete(next_ellipsoid, np.argmax(next_ellipsoid[:, 2]), axis=0)
        # next_ellipsoid = np.delete(next_ellipsoid, np.argmin(next_ellipsoid[:, 2]), axis=0)
        # Poly2 = Polygon(next_ellipsoid)
        # x, y = Poly2.exterior.xy
        # ax.plot(x, y)
        plt.show()
        # self.union = unary_union([Poly1, Poly2])
        return self.union
    
    def plot_ellips_union(self):
        x, y = self.union.exterior.xy
        plt.plot(x, y)
        plt.show()