import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from house import House

class Planner:

    def __init__(self, house: House, test_mode=False):
        self._house = house
        self._test_mode = test_mode

    def plan_motion(self, start=[0.,0.], end=[0.,0.]):
        """
        Plan the motion of the mobile manipulator with a starting position and a final position.
        @DISCLAIMER: Manually-written motion planning as of the moment.
        """
        MIN_CORNER, MAX_CORNER = self._house._corners

        def assert_coordinates(coord, type):
            assert MIN_CORNER[0] <= coord[0] <= MAX_CORNER[0], f"{type} x-position outside of expected range, got: {MIN_CORNER[0]} <= {start[0]} <= {MAX_CORNER[0]}"
            assert MIN_CORNER[1] <= coord[1] <= MAX_CORNER[1], f"{type} y-position outside of expected range, got: {MIN_CORNER[1]} <= {start[1]} <= {MAX_CORNER[1]}"

        assert_coordinates(start, 'Start')
        assert_coordinates(end, 'End')

        # Manually-written motion planning per room
        if not self._test_mode:
            self._routes = [
                [[-9.25,-3.5], [-9.25,-3.0], [-7.5,-3.0], [-6.5,-1.5]],
                [[-5.5,-1.5], [0.0,-1.5], [0.0, -2.5], [3.8, -3.5], [4.2,-4.5], [6.6,-4.2]],
                [[6.4,-3.5], [6.0,-1.9]],
            ]
        else:
            self._routes = [    # @TEST_MODE
                [start, end],
            ]

        # Manually-written doors' "openness" (ignore this)
        self._doors = [
            {
                'bathroom':         False,
                'outdoor':          False,
                'top_bedroom':      False,
                'bottom_bedroom':   False,
                'kitchen':          False,
            },
            {
                'bathroom':         False,
                'outdoor':          False,
                'top_bedroom':      False,
                'bottom_bedroom':   True,
                'kitchen':          False,
            },
            {
                'bathroom':         True,
                'outdoor':          False,
                'top_bedroom':      False,
                'bottom_bedroom':   True,
                'kitchen':          False,
            },
        ]

        # Initiation of motion planning
        no_rooms = len(self._routes)
        self.mp_done = False

        # Coordinates of obstacles
        self._lines, self._points, self._boxes = self._house.generate_plot_obstacles()
        obstacle_list = []
        for line in self._lines:
            obstacle = Obstacle(line['coord'][0], line['coord'][1])
            obstacle_list.append(obstacle)
        
        house_dim = [MIN_CORNER, MAX_CORNER]
        self.rrt = RRT(self._routes[0][0], self._routes[-1][-1], dim=house_dim, obstacle_list=obstacle_list, step_size=0.1, max_iter=5000)
        self.path = self.rrt.find_path()
        print(f'RRT: {len(self.path)}') if self.path is not None else print('RRT: 0')
        print(f'Vertices: {len(self.rrt.vertices)}')
        print(f'Vertices: {np.array(self.rrt.vertices)}')

        return no_rooms

    def generate_waypoints(self, room):
        assert len(self._routes[room]) > 0, f"There is no route generated. Run planner.plan_motion() before executing this method."
        assert len(self._doors[room]) > 0, f"There is no door 'openness' generated. Run planner.plan_motion() before executing this method."
        return self._routes[room], self._doors[room]

    def generate_trajectory(self, start, end, type=None):
        # TODO - Linear
        # TODO - Circular
        pass

    def plot_plan_2d(self, route):
        # Obtain the line and boxe coordinates of the walls, doors and furniture.
        # lines, points, boxes = self._house.generate_plot_obstacles()

        # Generate 2D plot of house.
        fig, ax = plt.subplots()

        # Plot the walls and doors as lines.
        for line in self._lines:
            x = np.array(line['coord'])[:,0]
            y = np.array(line['coord'])[:,1]
            if line['type'] == 'wall':  # Color walls as black
                color = 'black'
            else:                       # Color doors as green
                color = 'yellow'
            ax.plot(x,y, color, linewidth=2)

        # Plot the door knobs as points.
        for point in self._points:
            ax.plot(point[0], point[1], color='lime', marker='o', markersize=5)

        # Plot the furniture as boxes.
        for box in self._boxes:
            ax.add_patch(
                Rectangle((box['x'],box['y']),box['w'],box['h'],
                facecolor='blue',
                fill=True,
            ))

        # Plot the route as red vectors.
        for i in range(1,len(route)):
            x1 = route[i-1]
            x2 = route[i]
            magnitude_x = x2[0] - x1[0]
            magnitude_y = x2[1] - x1[1]
            theta = np.arctan2(magnitude_y, magnitude_x)
            ax.arrow(x1[0], x1[1], magnitude_x-0.25*np.cos(theta), magnitude_y-0.25*np.sin(theta), color='r', head_width=0.2, width=0.05)

        # Plot RRT
        for vertex in self.rrt.vertices:
            ax.plot(vertex[0], vertex[1], color='orange', marker='o', markersize=5)
        
        if len(self.path) > 0:
            for i in range(1,len(self.path)):
                x1 = self.path[i-1]
                x2 = self.path[i]
                magnitude_x = x2[0] - x1[0]
                magnitude_y = x2[1] - x1[1]
                theta = np.arctan2(magnitude_y, magnitude_x)
                ax.arrow(x1[0], x1[1], magnitude_x-0.25*np.cos(theta), magnitude_y-0.25*np.sin(theta), color='b', head_width=0.2, width=0.05)
        
        plt.show()

class RRT:
    def __init__(self, start, goal, dim, obstacle_list, step_size = 1.0, max_iter = 100):
        self.start = start
        self.goal = goal
        self.dim = dim
        self.obstacle_list = obstacle_list
        self.step_size = step_size
        self.max_iter = max_iter
        self.vertices = []
        self.edges = []

    def get_distance(self, point_1, point_2):
        """
        Helper function
        """
        return np.linalg.norm(np.array(point_2) - np.array(point_1))
        # return np.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)

    def get_heuristic(self, point):
        """
        Helper function
        """
        return self.get_distance(point[0:2], self.goal)

    def in_collision(self, point_1, point_2):
        """
        Helper function to check if a line segment between p1 and p2 is in collision
        """
        for obstacle in self.obstacle_list:
            if obstacle.check_collision(point_1, point_2):
                return True
        return False

    def find_nearest(self, point):
        """
        Helper function
        """
        min_dist = float('inf')
        nearest_point = None
        for vertex in self.vertices:
            print(f'min_dist {min_dist}')
            dist = self.get_distance(point, vertex[0:2])
            print(f'dist: {dist}')
            if dist < min_dist:
                nearest_point = vertex
                min_dist = dist
        
        return nearest_point
    
    def steer(self, random_point, nearest_point):
        """
        Helper function: steer
        """
        # if self.in_collision(nearest_point, random_point):
        #     return None
        new_point = random_point
        if self.get_distance(nearest_point, random_point) > self.step_size:
            new_point = np.array([
                nearest_point[0] + self.step_size*(random_point[0]-nearest_point[0])/self.get_distance(nearest_point,random_point),
                nearest_point[1] + self.step_size*(random_point[1]-nearest_point[1])/self.get_distance(nearest_point,random_point)
            ])
        return new_point

    def find_nearest_cluster(self, new_point):
        """
        Helper function
        """
        nearest_points = []
        for vertex in self.vertices:
            dist = self.get_distance(new_point, vertex[0:2])
            if dist < self.step_size:
                nearest_points.append(vertex)
        return nearest_points

    def choose_parent(self, new_point, nearest_point, nearest_points):
        """
        Helper function
        """
        min_cost = float(np.infty)
        chosen_parent = nearest_point
        for vertex in self.vertices:
            if vertex not in nearest_points:
                continue

            cost = self.get_distance(vertex[0:2], new_point[0:2])
            parent = vertex[2]
            while parent in nearest_points:
                next_parent = parent[2]
                print(f'parent: {parent}')
                print(f'next parent: {next_parent}')
                cost += self.get_distance(parent[0:2], next_parent[0:2])
                parent = next_parent

            if cost < min_cost:
                min_cost = cost
                parent = vertex
            
        return chosen_parent[0:2], min_cost


    def rewire(self, nearest_point, nearest_points):
        """
        Helper function
        """
        # for vertex in self.vertices:
        #     if self.in_collision(vertex[0:2], new_point[0:2]): # Ignore if the edge between the vertex and new node is not obstacle-free.
        #         continue
        #     if vertex[2] is None: # Ignore if the vertex is the starting node.
        #         continue
        #     cur_cost = self.dist(vertex[0:2], new_point[0:2]) + vertex[3]
        #     if cur_cost < vertex[3]:
        #         vertex[2] = 
        #         vertex[3] = cur_cost
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if vertex not in nearest_points: # Ignore vertices not in cluster
                continue
            
            if self.in_collision(vertex[0:2], nearest_point[0:2]): # Ignore vertices that are not obstacle free
                continue

            if vertex[2] is None: # Ignore starting vertex
                continue

            cost = nearest_point[3] + self.get_distance(vertex[0:2], nearest_point[0:2])
            if cost < vertex[3]:
                self.vertices[i][2] = nearest_point[0:2]
                self.vertices[i][3] = cost

    def find_path(self):
        min_x, min_y = self.dim[0]
        max_x, max_y = self.dim[1]
        self.vertices = [[self.start[0], self.start[1], None, 0.0]]
        while len(self.vertices) < self.max_iter:
            rand_point = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            nearest_point = self.find_nearest(rand_point)
            print(f'nearest point: {nearest_point}')
            new_point = self.steer(rand_point, nearest_point[0:2])

            if self.in_collision(new_point, nearest_point[0:2]):
                continue

            nearest_points = self.find_nearest_cluster(new_point[0:2])
            parent, cost = self.choose_parent(new_point, nearest_point, nearest_points)
            new_point = [new_point[0], new_point[1], parent, cost]
            self.vertices.append(new_point)
            self.rewire(nearest_point, nearest_points)
            
            if self.get_heuristic(new_point) < self.step_size:
                tree = [new_point]
                cur_point = new_point
                while cur_point[0:2] != self.start:
                    for vertex in self.vertices:
                        if vertex[0:2] == cur_point[2]:
                            cur_point = vertex
                            tree.append(vertex[0:2])
                            break
                return [tree[i][0:2] for i in range(len(tree))]
        return None

# class RRTStar:
#     def __init__(self, start, goal, dim, obstacle_list, rrt_star = True, step_size = 0.5, max_iter = 100):
#         self.start = np.array(start)
#         self.goal = np.array(goal)
#         self.dim = dim
#         self.obstacle_list = obstacle_list
#         self.rrt_star = rrt_star
#         self.step_size = step_size
#         self.max_iter = max_iter
#         self.trees = []
        
#     def dist(self, p1, p2):
#         """
#         Helper function to calculate the distance between two points
#         """
#         return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
#         # return np.linalg.norm(p2-p1)

#     def heuristic(self, p):
#         """
#         Helper function to calculate the heuristic between a point and the end goal.
#         """
#         return self.dist(self.goal, p)
    
#     def nearest(self, point):
#         """
#         Helper function to find the nearest vertex to a given point
#         """
#         min_dist = float('inf')
#         nearest_vertex = None
#         parent_idx = None
#         for idx, vertex in enumerate(self.trees):
#             vertex = np.array(vertex)
#             dist = self.dist(vertex[0:2], point)
#             if dist < min_dist:
#                 min_dist = dist
#                 nearest_vertex = vertex
#                 parent_idx = idx
#         return np.array(nearest_vertex), parent_idx
    
#     def in_collision(self, p1, p2):
#         """
#         Helper function to check if a line segment between p1 and p2 is in collision
#         """
#         for obstacle in self.obstacle_list:
#             if obstacle.check_collision(p1, p2):
#                 return True
#         return False
        
#     def extend(self, p1, p2):
#         """
#         Helper function to extend the tree towards a point
#         """
#         if self.in_collision(p1, p2):
#             return None
#         if self.dist(p1, p2) > self.step_size:
#             p2 = np.array([
#                 p1[0] + self.step_size*(p2[0]-p1[0])/self.dist(p1,p2),
#                 p1[1] + self.step_size*(p2[1]-p1[1])/self.dist(p1,p2)
#             ])
#         return p2
    
#     def rewire(self, new_point, parent_idx):
#         """Rewire the tree to improve the path"""
#         for point in self.trees:
#             point = np.array(point)
#             if self.in_collision(new_point[0:2], point[0:2]):
#                 continue
#             if point[2] is None:
#                 continue
#             cur_cost = self.dist(point, new_point) + new_point[3]
#             if cur_cost < point[3]:
#                 point[2] = parent_idx
#                 point[3] = cur_cost

#     def find_path(self):
#         """Function to find the path from start to goal"""
#         min_x, min_y = self.dim[0]
#         max_x, max_y = self.dim[1]
#         self.trees = [[self.start[0], self.start[1], None, 0]]
#         while len(self.trees) < self.max_iter:
#             rand_point = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
#             nearest_point, parent_idx = self.nearest(rand_point)
#             new_point = self.extend(nearest_point[0:2], rand_point)
            
#             if self.in_collision(new_point, nearest_point[0:2]):
#                 continue

#             new_point = [new_point[0], new_point[1], len(self.trees)-1, self.dist(new_point, nearest_point[0:2]) + self.trees[len(self.trees)-1][3]]
#             self.trees.append(new_point)

#             if self.rrt_star:
#                 self.rewire(np.array(new_point), parent_idx)

#             if self.heuristic(new_point[0:2]) <= self.step_size:
#                 path = [new_point[0:2]]
#                 cur_point = new_point
#                 while cur_point[0:2] != self.start.tolist():
#                     for idx, point in enumerate(self.trees):
#                         if cur_point[2] == idx:
#                             cur_point = point
#                             path.append(point[0:2])
#                             break
#                 # return [path[i][:2] for i in range(len(path))][::-1]
#                 return path
#         return None

class Obstacle:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        
    def check_collision(self, p1, p2):
        """ Check if the line segment from p1 to p2 intersects with the obstacle"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.v1
        x4, y4 = self.v2
        
        denominator = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        if denominator == 0:
            return False
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denominator
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denominator
        
        if (0 <= t <= 1) and (0 <= u <= 1):
            return True
        return False
