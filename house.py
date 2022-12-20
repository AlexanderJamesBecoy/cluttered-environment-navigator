import gym
from door import Door
# import pybullet as p
import numpy as np
from ObstacleConstraintGenerator import ObstacleConstraintsGenerator

HEIGHT = 0.5
WIDTH = 0.3

class House:
# This class contains all the useful, but approximated, information that describes a house for which a mobile manipulator
# will explore. It will contain the goal objects which are the door knobs that is our mobile manipulator will try to reach.
# Further details of class House:
# - Walls
# - Doors
# - Furniture

    def __init__(self, env, robot_dim: list, scale: float):
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
        self.Obstacles = ObstacleConstraintsGenerator(robot_dim=robot_dim, scale=scale)

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
        
        self.Obstacles.walls = np.array(self.Obstacles.walls)

    def add_wall(self, start_pos, end_pos, wall_thickness=0.1, wall_height=0.5):
    # This function draws a wall segment into gym `env` from a starting position `start_pos` to a final position `end_pos`.
    # The default thickness `wall_thickness` and `wall_height` are 10 cm and 50 cm, respectively. They are modifiable.

        vec = end_pos - start_pos       # Obtain a vector from the two points.
        avg = (end_pos + start_pos)/2   # Obtain the average point between the two points, because
                                        # gym `env` draws the shape centered.
        theta = np.arctan2(*vec)        # Obtain the angle of the vector.
    
        dim = np.array([WIDTH, np.linalg.norm(vec), HEIGHT])    # Obtain the dimension of the wall.
        pos = [[avg[0], avg[1], theta]]                         # Describe the position of the wall with average position and angle.
        self.Obstacles.walls.append({'x': pos[0][0], 'y': pos[0][1], 'theta': pos[0][2], 'width': dim[0], 'length': dim[1], 'height': dim[2]}) # Add new obstacle pos to list
        self.env.add_shapes(shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos)

    def generate_doors(self):
        # Add all door and door knobs to pos list and convert all lists to np arrays
        door_bathroom = Door(self.env, pos=self.points['W']-self.offset, is_open=True, theta=0, is_flipped=True)
        door_bathroom.draw_door()
        self.Obstacles.doors.append({'x': door_bathroom.pos_door[0][0], 'y': door_bathroom.pos_door[0][1], 'theta': door_bathroom.pos_door[0][2], 
                                        'width': door_bathroom.dim_door[0], 'length': door_bathroom.dim_door[1], 'height': door_bathroom.dim_door[2]})
        self.Obstacles.knobs.append({'x': door_bathroom.pos_knob[0][0], 'y': door_bathroom.pos_knob[0][1], 'theta': door_bathroom.pos_knob[0][2], 
                                        'width': door_bathroom.dim_knob[0], 'length': door_bathroom.dim_knob[1], 'height': door_bathroom.dim_knob[2]})

        door_outdoor = Door(self.env, pos=self.points['E']-self.offset, is_open=True, theta=np.pi)
        door_outdoor.draw_door()
        self.Obstacles.doors.append({'x': door_outdoor.pos_door[0][0], 'y': door_outdoor.pos_door[0][1], 'theta': door_outdoor.pos_door[0][2], 
                                        'width': door_outdoor.dim_door[0], 'length': door_outdoor.dim_door[1], 'height': door_outdoor.dim_door[2]})
        self.Obstacles.knobs.append({'x': door_outdoor.pos_knob[0][0], 'y': door_outdoor.pos_knob[0][1], 'theta': door_outdoor.pos_knob[0][2], 
                                        'width': door_outdoor.dim_knob[0], 'length': door_outdoor.dim_knob[1], 'height': door_outdoor.dim_knob[2]})

        door_bedroom1 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0)
        door_bedroom1.draw_door()
        self.Obstacles.doors.append({'x': door_bedroom1.pos_door[0][0], 'y': door_bedroom1.pos_door[0][1], 'theta': door_bedroom1.pos_door[0][2], 
                                        'width': door_bedroom1.dim_door[0], 'length': door_bedroom1.dim_door[1], 'height': door_bedroom1.dim_door[2]})
        self.Obstacles.knobs.append({'x': door_bedroom1.pos_knob[0][0], 'y': door_bedroom1.pos_knob[0][1], 'theta': door_bedroom1.pos_knob[0][2], 
                                        'width': door_bedroom1.dim_knob[0], 'length': door_bedroom1.dim_knob[1], 'height': door_bedroom1.dim_knob[2]})

        door_bedroom2 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0.5*np.pi, is_flipped=True)
        door_bedroom2.draw_door()
        self.Obstacles.doors.append({'x': door_bedroom2.pos_door[0][0], 'y': door_bedroom2.pos_door[0][1], 'theta': door_bedroom2.pos_door[0][2], 
                                        'width': door_bedroom2.dim_door[0], 'length': door_bedroom2.dim_door[1], 'height': door_bedroom2.dim_door[2]})
        self.Obstacles.knobs.append({'x': door_bedroom2.pos_knob[0][0], 'y': door_bedroom2.pos_knob[0][1], 'theta': door_bedroom2.pos_knob[0][2], 
                                        'width': door_bedroom2.dim_knob[0], 'length': door_bedroom2.dim_knob[1], 'height': door_bedroom2.dim_knob[2]})

        door_kitchen = Door(self.env, pos=self.points['I']-self.offset, is_open=True, theta=-0.5*np.pi, is_flipped=True)
        door_kitchen.draw_door()
        self.Obstacles.doors.append({'x': door_kitchen.pos_door[0][0], 'y': door_kitchen.pos_door[0][1], 'theta': door_kitchen.pos_door[0][2], 
                                        'width': door_kitchen.dim_door[0], 'length': door_kitchen.dim_door[1], 'height': door_kitchen.dim_door[2]})
        self.Obstacles.knobs.append({'x': door_kitchen.pos_knob[0][0], 'y': door_kitchen.pos_knob[0][1], 'theta': door_kitchen.pos_knob[0][2], 
                                        'width': door_kitchen.dim_knob[0], 'length': door_kitchen.dim_knob[1], 'height': door_kitchen.dim_knob[2]})
        self.Obstacles.doors = np.array(self.Obstacles.doors)
        self.Obstacles.knobs = np.array(self.Obstacles.knobs)

