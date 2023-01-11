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
                [[-3.0,-3.0], [3.0,3.0]],
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

        # Initialize starting position
        start_pos = self._routes[0][0]

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
        lines, points, boxes = self._house.generate_plot_obstacles()

        # Generate 2D plot of house.
        fig, ax = plt.subplots()

        # Plot the walls and doors as lines.
        for line in lines:
            x = np.array(line['coord'])[:,0]
            y = np.array(line['coord'])[:,1]
            if line['type'] == 'wall':  # Color walls as black
                color = 'black'
            else:                       # Color doors as green
                color = 'yellow'
            ax.plot(x,y, color, linewidth=2)

        # Plot the door knobs as points.
        for point in points:
            ax.plot(point[0], point[1], color='lime', marker='o', markersize=5)

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