import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from house import House
from model import Model

class Planner:

    def __init__(self, robot: Model, house: House):
        self._robot = robot
        self._house = house

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
        self._routes = [
            # [[0, -2], [2, -2], [2, 0], [0, 0], [0, 10], [10, 10], [-10, -10]],
            [[-9.25,-3.5], [-9.25,-3.0], [-7.5,-3.0], [-6.5,-1.5]],
        ]

        # Manually-written doors' "openness"
        self._doors = [
            {
                'bathroom':         False,
                'outdoor':          False,
                'top_bedroom':      False,
                'bottom_bedroom':   False,
                'kitchen':          False,
            },
        ]

        # Initiation of motion planning
        self._done = False

        # Initialize starting position
        start_pos = self._routes[0][0]
        return start_pos, self._doors[0]

    def generate_waypoints(self):
        # self._waypoints = [0, -2], [2, -2], [2, 0], [0, 0], [0, 10], [10, 10], [-10, -10]
        return self._routes[0]

    def generate_trajectory(self, start, end, type=None):
        # TODO - Linear
        # TODO - Circular
        pass

    def plot_plan_2d(self):
        route = self._routes[0]
        print(route)

        # Obtain the line and boxe coordinates of the walls, doors and furniture.
        lines, boxes = self._house.generate_plot_obstacles()

        # Generate 2D plot of house.
        fig, ax = plt.subplots()

        # Plot the walls and doors as lines.
        for line in lines:
            x = np.array(line['coord'])[:,0]
            y = np.array(line['coord'])[:,1]
            if line['type'] == 'wall':  # Color walls as black
                color = 'black'
            else:                       # Color doors as green
                color = 'lime'
            ax.plot(x,y, color, linewidth=2)

        # Plot the furniture as boxes.
        for box in boxes:
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