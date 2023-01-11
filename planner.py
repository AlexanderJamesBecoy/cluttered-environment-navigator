import random
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
        rrt = RRTStar(self._routes[0][0], self._routes[0][1], dim=house_dim, obstacle_list=obstacle_list, rrt_star=True, step_size=1.0, max_iter=100)
        path = rrt.find_path()
        print(f'RRT: {path}')

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
        
        plt.show()

class RRTStar:
    def __init__(self, start, goal, dim, obstacle_list, rrt_star = True, step_size = 1, max_iter = 1000):
        self.start = start
        self.goal = goal
        self.dim = dim
        self.obstacle_list = obstacle_list
        self.rrt_star = rrt_star
        self.step_size = step_size
        self.max_iter = max_iter
        self.vertices = []
        self.vertices.append(start)
        
    def dist(self, p1, p2):
        """
        Helper function to calculate the distance between two points
        """
        # return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return np.linalg.norm(np.array(p2)-np.array(p1))
    
    def nearest(self, point):
        """
        Helper function to find the nearest vertex to a given point
        """
        min_dist = float('inf')
        nearest_vertex = None
        for vertex in self.vertices:
            dist = self.dist(vertex, point)
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = vertex
        return nearest_vertex
    
    def in_collision(self, p1, p2):
        """
        Helper function to check if a line segment between p1 and p2 is in collision
        """
        for obstacle in self.obstacle_list:
            if obstacle.check_collision(p1, p2):
                return True
        return False
        
    def extend(self, p1, p2):
        """
        Helper function to extend the tree towards a point
        """
        if self.in_collision(p1, p2):
            return None
        if self.dist(p1, p2) > self.step_size:
            p2 = (p1[0] + self.step_size*(p2[0]-p1[0])/self.dist(p1,p2),
                  p1[1] + self.step_size*(p2[1]-p1[1])/self.dist(p1,p2))
        return p2
    
    def rewire(self, new_point, tree):
        """Rewire the tree to improve the path"""
        for point in tree:
            if self.in_collision(new_point[0:2], point[0:2]):
                continue
            if point[2] is None:
                continue
            cur_cost = self.dist(point, new_point) + new_point[3]
            if cur_cost < point[3]:
                point[2] = len(tree)-1
                point[3] = cur_cost

    def find_path(self):
        """Function to find the path from start to goal"""
        min_x, min_y = self.dim[0]
        max_x, max_y = self.dim[1]
        tree = [(self.start[0], self.start[1], None, 0)]
        while len(tree) < self.max_iter:
            rand_point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            print(f'rand_point: {rand_point}')
            self.vertices.append(rand_point)
            nearest_point = self.nearest(rand_point)
            new_point = self.extend(nearest_point, rand_point)
            if self.in_collision(new_point, nearest_point):
                continue

            new_point = (new_point[0], new_point[1], len(tree)-1, self.dist(new_point, nearest_point) + tree[len(tree)-1][3])
            tree.append(new_point)

            if self.rrt_star:
                self.rewire(new_point, tree)

            if self.dist(new_point[0:2], self.goal) <= self.step_size:
                path = [new_point]
                cur_point = new_point
                while cur_point != self.start:
                    for point in tree:
                        if point[2] == cur_point[2]:
                            cur_point = point
                            path.append(point)
                            break
                return [path[i][:2] for i in range(len(path))][::-1]
        print(f'tree: {np.array(tree).shape}')
        return None

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
