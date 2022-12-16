import gym
from door import Door
import numpy as np

class House:
# This class contains all the useful, but approximated, information that describes a house for which a mobile manipulator
# will explore. It will contain the goal objects which are the door knobs that is our mobile manipulator will try to reach.
# Further details of class House:
# - Walls
# - Doors
# - Furniture

    def __init__(self, env):
    # Initialize an object of this class. It only requires the pointer to the gym environment `env`.
    # Following declarations: 
    # - `offset` to center the points around the origin (0,0).
    # - `points` to describe the walls and door hinges (also a vertex describing the end of wall segment).
    # - `doors` is a list of interactable door objects.

        self.env = env
        self.offset = np.array([7.0, 3.5])
        self.points = {
            'A': np.array([0.0,0.0]),   # Wall vertex.
            'B': np.array([12.0,0.0]),  # Wall vertex.
            'C': np.array([0.0,7.0]),   # Wall vertex.
            'D': np.array([7.0,7.0]),   # Wall vertex.
            'E': np.array([8.0,7.0]),   # Wall vertex / Door hinge to outside.
            'F': np.array([11.0,7.0]),  # Wall vertex.
            'G': np.array([11.0,3.0]),  # Wall vertex.
            'H': np.array([12.0,3.0]),  # Wall vertex.
            'I': np.array([8.0,3.0]),   # Wall vertex / Door hinge to the kitchen.
            'J': np.array([5.0,7.0]),   # Wall vertex.
            'K': np.array([5.0,3.0]),   # Wall vertex.
            'L': np.array([6.0,7.0]),   # Wall vertex.
            'M': np.array([6.0,3.0]),   # Wall vertex.
            'N': np.array([8.0,4.0]),   # Wall vertex.
            'O': np.array([0.0,3.0]),   # Wall vertex.
            'P': np.array([3.0,3.0]),   # Wall vertex / Door hinges to the two bedrooms.
            'Q': np.array([3.0,0.0]),   # Wall vertex.
            'R': np.array([3.0,2.0]),   # Wall vertex.
            'S': np.array([10.0,3.0]),  # Wall vertex.
            'T': np.array([10.0,1.0]),  # Wall vertex.
            'U': np.array([11.0,1.0]),  # Wall vertex.
            'V': np.array([4.0,3.0]),   # Wall vertex.
            'W': np.array([12.0,1.0]),  # Wall vertex / Door hinge to the bathroom.
        }
        self.doors = []

    def generate_walls(self):
    # This function generates and draw the fixed wall segments described in `self.points`.

        points = np.array([ # Generate wall edges
            [self.points['A'], self.points['B']],
            [self.points['A'], self.points['C']],
            [self.points['C'], self.points['D']],
            [self.points['E'], self.points['F']],
            [self.points['F'], self.points['G']],
            [self.points['B'], self.points['H']],
            [self.points['I'], self.points['H']],
            [self.points['J'], self.points['K']],
            [self.points['V'], self.points['M']],
            [self.points['L'], self.points['M']],
            [self.points['E'], self.points['N']],
            [self.points['O'], self.points['P']],
            [self.points['Q'], self.points['R']],
            [self.points['S'], self.points['T']],
            [self.points['T'], self.points['U']],
        ])

        for i in range(points.shape[0]):
        # Iterate for every wall edge, center the wall vertices with `self.offset`, and draw the wall.
            start_pos = points[i][0] - self.offset
            end_pos = points[i][1] - self.offset
            self.add_wall(start_pos, end_pos)

    def add_wall(self, start_pos, end_pos, wall_thickness=0.3, wall_height=0.5):
    # This function draws a wall segment into gym `env` from a starting position `start_pos` to a final position `end_pos`.
    # The default thickness `wall_thickness` and `wall_height` are 30 cm and 50 cm, respectively. They are modifiable.

        vec = end_pos - start_pos       # Obtain a vector from the two points.
        avg = (end_pos + start_pos)/2   # Obtain the average point between the two points, because
                                        # gym `env` draws the shape centered.
        theta = np.arctan2(*vec)        # Obtain the angle of the vector.
    
        dim = np.array([wall_thickness, np.linalg.norm(vec), wall_height])  # Obtain the dimension of the wall.
        pos = [[avg[0], avg[1], theta]]                                     # Describe the position of the wall with average position and angle.
        self.env.add_shapes(shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos)   # Draw the wall as a geometrical box.

    def generate_doors(self):
        door_bathroom = Door(self.env, pos=self.points['W']-self.offset, is_open=True, theta=0, is_flipped=True)
        door_bathroom.draw_door()
        door_outdoor = Door(self.env, pos=self.points['E']-self.offset, is_open=True, theta=np.pi)
        door_outdoor.draw_door()
        door_bedroom1 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0)
        door_bedroom1.draw_door()
        door_bedroom2 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0.5*np.pi, is_flipped=True)
        door_bedroom2.draw_door()
        door_kitchen = Door(self.env, pos=self.points['I']-self.offset, is_open=True, theta=-0.5*np.pi, is_flipped=True)
        door_kitchen.draw_door()


