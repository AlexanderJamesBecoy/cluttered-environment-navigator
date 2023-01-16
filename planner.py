import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from house import House

class Planner:
    """
    This class is dedicated on creating a motion planning for the robot within a given house.
    It is a sampling-based planner that implements RRT*.
    """

    def __init__(self, house: House, test_mode=False, debug_mode=False, doors_exist=True, door_opens=False):
        """
        @param house        - store the pointer to House object.
        @param test_mode    - obtain the test house; simple box to test the mobile manipulator functionality.
        @param debug_mode   - let this object print data of motion planning in terminal.
        @param doors        - boolean to set if doors exist.
        @param door_opens   - make all door opens at beginning if this is True.
        """
        self._house = house
        self._test_mode = test_mode
        self._debug_mode = debug_mode
        self._doors_exist = doors_exist
        self._door_opens = door_opens

    def plan_motion(self, start=[0.,0.], end=[0.,0.], step_size=0.5, max_iter=1000):
        """
        Plan the motion of the mobile manipulator with a starting position and a final position.
        @start      - starting position in 2D
        @end        - end position in 2D
        @step_size  - set the maximum size between two vertices interval
        @max_iter   - set the maximal number of random samples

        Returns the number of rooms
        """
        # Obtain the minimal and maximal XY-coordinate values of the house.
        MIN_CORNER, MAX_CORNER = self._house._corners
        house_dim = [MIN_CORNER, MAX_CORNER]

        #Assert if `start` and `end` is within the house.
        def assert_coordinates(coord, type):
            assert MIN_CORNER[0] <= coord[0] <= MAX_CORNER[0], f"{type} x-position outside of expected range, got: {MIN_CORNER[0]} <= {start[0]} <= {MAX_CORNER[0]}"
            assert MIN_CORNER[1] <= coord[1] <= MAX_CORNER[1], f"{type} y-position outside of expected range, got: {MIN_CORNER[1]} <= {start[1]} <= {MAX_CORNER[1]}"

        assert_coordinates(start, 'Start')
        assert_coordinates(end, 'End')

        # Obtain the lines (walls), points and boxes.
        self._lines, self._points, self._boxes = self._house.generate_plot_obstacles(door_generated=False)
        obstacle_list = [] # Create a list of Obstacle objects.

        # Generate an Obstacle object and append into list for every line segment
        for line in self._lines:
            if line['type'] == 'door':
                continue
            obstacle = Obstacle(line['coord'][0], line['coord'][1])
            obstacle_list.append(obstacle)

        # Generate four Obstacle objects and append into list for every line segment in a box
        for box in self._boxes:
            # Ignore furniture that are suspended in the air.
            if box['floating']:
                continue

            # Obtain the coordinates of the box in XY-plane.
            x1 = box['x']
            y1 = box['y']
            x2 = x1 + box['w']
            y2 = y1 + box['h']

            # Create an Obstacle object and append into list
            obstacle_left = Obstacle([x1,y1], [x1,y2])
            obstacle_right = Obstacle([x2,y1], [x2,y2])
            obstacle_up = Obstacle([x1,y1], [x2,y1])
            obstacle_down = Obstacle([x1,y2], [x2,y2])
            obstacle_list.append(obstacle_left)
            obstacle_list.append(obstacle_right)
            obstacle_list.append(obstacle_up)
            obstacle_list.append(obstacle_down)
    
        # Start measuring the RRT* computation time
        if self._debug_mode:
            start_time = time.time()

        # Create a RRT object and start finding a path.
        self.rrt = RRT(start=start, goal=end, dim=house_dim, obstacle_list=obstacle_list, step_size=step_size, max_iter=max_iter, debug_mode=self._debug_mode)
        self.path, path_cost = self.rrt.find_path()
        # Assert if path is found.
        assert self.path is not None, f"There is no optimal path found with RRT* with parameters `step_size` {step_size} and `max_iter` {max_iter}. Please restart the simulation or adjust the parameters."
        
        # Obtain a list of room that the robot will explore.
        room_history = []
        self._routes = []
        bifurcation_idx = 0
        room = self._house.get_room(self.path[0][0], self.path[0][1])
        room_history.append(room)
        for vertex_idx, vertex in enumerate(self.path[1:]):
            room = self._house.get_room(vertex[0], vertex[1])
            if room is None: # Ignore if vertex's room has no unique door
                continue
            if room == room_history[-1]: # Ignore if vertex's room is same as previous.
                continue
            route = self.path[bifurcation_idx:vertex_idx+1]
            bifurcation_idx = vertex_idx + 1
            room_history.append(room)
            self._routes.append(route)
        route = self.path[bifurcation_idx:]
        self._routes.append(route)
        self._doors = [self._house._doors_open.copy()]
        for room in room_history:
            if self._house._doors_open[room] is None:
                continue
            self._house._doors_open[room] = True
            self._doors.append(self._house._doors_open.copy())

        # Print the information of sampling-based planner implementation if `debug_mode` is activated.
        if self._debug_mode:
            print(f'RRT: {len(self.path)}') if self.path is not None else print('RRT: 0')
            print(f'Vertices: {len(self.rrt.vertices)}')
            print(f'Cost: {path_cost} m')
            print(f'RRT execution time: {round(time.time() - start_time,3)} s')
            print(f'Room exploration: {room_history}')
            print(f'Doors: {self._doors}')

        return len(self._routes)

    def generate_waypoints(self, room):
        """
        Return a list of points describing a path within a room; given the room number.
        """
        # assert len(self._routes[room]) > 0, f"There is no route generated. Run planner.plan_motion() before executing this method."
        # assert len(self._doors[room]) > 0, f"There is no door 'openness' generated. Run planner.plan_motion() before executing this method."
        return self._routes[room], self._doors[room]

    def plot_plan_2d(self, room_idx=None):
        """
        Make a 2D plot of the house and the found path given the room number.
        """

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
        if self._doors_exist:
            for point in self._points:
                ax.plot(point[0], point[1], color='lime', marker='o', markersize=5)

        # Plot the furniture as boxes.
        for box in self._boxes:
            opacity = 1.0
            if box['floating']:
                opacity = 0.4
            ax.add_patch(
                Rectangle((box['x'],box['y']),box['w'],box['h'],
                facecolor='blue',
                fill=True,
                alpha=opacity,
            ))

        # Plot RRT* tree as gray lines
        for vertex in self.rrt.vertices:
            if vertex[3] is None:
                continue
            parent = self.rrt.vertices[vertex[3]]
            x = [parent[1], vertex[1]]
            y = [parent[2], vertex[2]]
            ax.plot(x, y, color='gray', alpha=0.6, linewidth=1)
        
        # Plot the route as red vectors.
        if self.path is not None:
            for i in range(1,len(self.path)):
                x1 = self.path[i-1]
                x2 = self.path[i]
                magnitude_x = x2[0] - x1[0]
                magnitude_y = x2[1] - x1[1]
                ax.arrow(x1[0], x1[1], magnitude_x*0.9, magnitude_y*0.9, color='r', head_width=0.2, width=0.01)
                # circle = plt.Circle((x1[0], x1[1]), self.rrt.step_size, color='orange', fill=False)
                # ax.add_patch(circle)

        # Plot the route in the room as green vectors.
        if self._doors_exist and room_idx is not None:
            for i in range(1,len(self._routes[room_idx])):
                x1 = self._routes[room_idx][i-1]
                x2 = self._routes[room_idx][i]
                magnitude_x = x2[0] - x1[0]
                magnitude_y = x2[1] - x1[1]
                theta = np.arctan2(magnitude_y, magnitude_x)
                ax.arrow(x1[0], x1[1], 0.8*magnitude_x*np.cos(theta), 0.8*magnitude_y*np.sin(theta), color='g', head_width=0.2, width=0.05)
            plt.title(f'RRT* implementation on route {room_idx+1}/{len(self._routes)}')
        else:
            plt.title('RRT* implementation')
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()

class RRT:
    """
    This class is a sampling-based planner based on RRT* method. Adaptable to any area of class House.
    """

    def __init__(self, start, goal, dim, obstacle_list, step_size=1.0, max_iter=100, debug_mode=False):
        """
        Store the arguments and create a list of vertices.
        @param start            - set the starting position
        @param end              - set the final position
        @param dim              - store the minimal and maximal XY-coordinate values of the house.
        @param obstacle_list    - list of Obstacle objects
        @step_size              - set the maximum size between two vertices interval
        @max_iter               - set the maximal number of random samples
        @param debug_mode   - let this object print data of motion planning in terminal. 
        """
        self.start = start
        self.goal = goal
        self.dim = dim
        self.obstacle_list = obstacle_list
        self.step_size = step_size
        self.max_iter = max_iter
        self.vertices = []
        self.debug_mode = debug_mode

    def get_distance(self, point_1, point_2):
        """
        Obtain the Euclidean distance between two 2D points.
        Returns float.
        """
        return np.linalg.norm(np.array(point_2) - np.array(point_1))

    def get_heuristic(self, point):
        """
        Obtain the Euclidean distance between a point and the final position.
        Returns float.
        """
        return self.get_distance(point[1:3], self.goal)

    def in_collision(self, point_1, point_2):
        """
        Return boolean if a line segment between `point_1` and `point_2` is in collision
        """
        random.shuffle(self.obstacle_list)  # Shuffle the list of Obstacle objects.
        for obstacle in self.obstacle_list:
            if obstacle.check_collision(point_1, point_2):
                return True
        return False

    def find_nearest(self, point):
        """
        Find the nearest vertex in `self.vertices` to the newly generated `point`.
        Return a vertex: [index, x_pos, y_pos, parent, cost]
        """
        min_dist = float('inf')
        nearest_point = None
        for vertex in self.vertices:
            dist = self.get_distance(point, vertex[1:3])
            if dist < min_dist:
                nearest_point = vertex
                min_dist = dist
        
        return nearest_point
    
    def steer(self, random_point, nearest_point, option='default'):
        """
        Adjust the distance between the two points if its larger than the step_size.
        Return new_point with a distance of step_size to `nearest_point`.
        """
        # Check if the nearest point and the newly-generated point make a collision with an obstacle.
        if self.in_collision(nearest_point, random_point):
            return None

        if self.get_distance(nearest_point, random_point) > self.step_size:
            if option == 'random':
                theta = np.random.uniform(-np.pi, np.pi)
                random_point = [
                    nearest_point[0] + self.step_size*np.cos(theta),
                    nearest_point[1] + self.step_size*np.sin(theta),
                ]
            else:
                random_point = [
                    nearest_point[0] + self.step_size*(random_point[0]-nearest_point[0])/self.get_distance(nearest_point,random_point),
                    nearest_point[1] + self.step_size*(random_point[1]-nearest_point[1])/self.get_distance(nearest_point,random_point),
                ]
        return random_point

    def find_nearest_cluster(self, new_point):
        """
        Return the indices of existing vertices that is within the distance of step_size.
        """
        nearest_points = []
        for vertex in self.vertices:
            dist = self.get_distance(new_point, vertex[1:3])
            if dist < self.step_size:
                nearest_points.append(vertex[0])
        return nearest_points

    def choose_parent(self, new_point, nearest_point, nearest_points):
        """
        Choose the parent of `new_point` that results in minimal cost.
        Return the parent's index and the cost.
        """
        # Set the initial parent to be the nearest point
        chosen_parent = nearest_point[0]
        min_cost = nearest_point[4] + self.get_distance(new_point, nearest_point[1:3])
        for vertex_idx in nearest_points:
            # Ignore the initial nearest_point
            if vertex_idx == nearest_point[0]:
                continue
            vertex = self.vertices[vertex_idx]
            cost = vertex[4] + self.get_distance(new_point, vertex[1:3])
            # Replace the parent if the cost is smaller
            if cost < min_cost:
                chosen_parent = vertex[0]
                min_cost = cost
        return chosen_parent, min_cost

    def rewire(self, new_point, nearest_points):
        """
        Rewire the structure of the nearest_points near new_point depending on the cost.
        """
        for vertex_idx in nearest_points:
            vertex = self.vertices[vertex_idx]
            if self.in_collision(vertex[1:3], new_point[1:3]):  # Ignore line segments that result in obstacle collision.
                continue
            if vertex[3] is None:                               # Ignore the starting node.
                continue
            cost = new_point[4] + self.get_distance(vertex[1:3], new_point[1:3])
            # Rewire the vertex to the new point
            if cost < vertex[4]:
                self.vertices[vertex_idx][3] = new_point[0]
                self.vertices[vertex_idx][4] = cost

    def find_path(self):
        """
        RRT* implementation: 
        Returns path (as list of points), total cost
        """
        # Obtain the minimal and maximal XY-values of the house. This is used for uniform-random samples.
        min_x, min_y = self.dim[0]
        max_x, max_y = self.dim[1]

        # Append the starting position into the list.
        self.vertices = [[0, self.start[0], self.start[1], None, 0.0]]

        # Iterate until max number of samples.
        while len(self.vertices) < self.max_iter:
            # Create a random sample, find the nearest vertex and steer the new function.
            rand_point = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            nearest_point = self.find_nearest(rand_point)
            new_point = self.steer(random_point=rand_point, nearest_point=nearest_point[1:3], option='default')

            # Print the nearest point if debug_mode is activated.
            if self.debug_mode:
                print(f'nearest point to point {len(self.vertices)}: {nearest_point}')

            # Ignore if the new point results in an obstacle collision.
            if new_point is None:
                continue

            # Ignore if the new point results in an obstacle collision.
            if self.in_collision(new_point, nearest_point[1:3]):
                continue

            # Set index, x_pos, y_pos, parent, cost to `new_point` and append into list of vertices.
            i = len(self.vertices)
            nearest_points_idx = self.find_nearest_cluster(new_point)
            parent, cost = self.choose_parent(new_point, nearest_point, nearest_points_idx)
            new_point = [i, new_point[0], new_point[1], parent, cost]
            self.vertices.append(new_point)
            if self.debug_mode:
                print(f'new point {len(self.vertices)-1}: {new_point}')

            # Rewire the nearest points to new_point in the tree structure
            self.rewire(new_point, nearest_points_idx)
            
            # Determine if new_point is close to the goal.
            if self.get_heuristic(new_point) <= self.step_size:
                path = [self.goal, new_point[1:3]]
                cur_point = new_point
                while cur_point[1:3] != self.start:
                    dead_end = True
                    for vertex in self.vertices:
                        if vertex[0] == cur_point[3]:
                            print(f'Append vertex to path: {vertex[1:3]}, length: {len(path)}')
                            cur_point = vertex
                            path.append(vertex[1:3])
                            dead_end = False
                            break
                    # Break pathfinding due to loops
                    if dead_end:
                        break
                # Return found path
                if cur_point[1:3] == self.start:
                    return path[::-1], new_point[4]
        # No path is found
        return None, 0

class Obstacle:
    """
    This class creates an object representing a line obstacle given the two 2D vertices. This object is then used for RRT*.
    """

    def __init__(self, vertex_1, vertex_2):
        """
        Store the vertices that describe the line obstacle.
        """
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        
    def check_collision(self, point_1, point_2):
        """
        Returns boolean if the line segment from `point_1` to `point_2` intersects with the obstacle described by `vertex_1` and `vertex_2`.
        """
        x1, y1 = point_1
        x2, y2 = point_2
        x3, y3 = self.vertex_1
        x4, y4 = self.vertex_2
        
        denominator = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        if denominator == 0:
            return False
        
        t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denominator
        u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denominator
        
        if (0 <= t <= 1) and (0 <= u <= 1):
            return True
        return False
