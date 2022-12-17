import gym
from door import Door
import numpy as np
from ObstacleConstraintGenerator import ObstacleConstraintsGenerator

HEIGHT = 0.5
WIDTH = 0.3

class House:
    def __init__(self, env, robot_dim: list, scale: float):
        self.env = env
        self.offset = np.array([7.0, 3.5])
        self.points = {
            'A': np.array([0.0,0.0]),
            'B': np.array([12.0,0.0]),
            'C': np.array([0.0,7.0]),
            'D': np.array([7.0,7.0]),
            'E': np.array([8.0,7.0]),
            'F': np.array([11.0,7.0]),
            'G': np.array([11.0,3.0]),
            'H': np.array([12.0,3.0]),
            'I': np.array([8.0,3.0]),
            'J': np.array([5.0,7.0]),
            'K': np.array([5.0,3.0]),
            'L': np.array([6.0,7.0]),
            'M': np.array([6.0,3.0]),
            'N': np.array([8.0,4.0]),
            'O': np.array([0.0,3.0]),
            'P': np.array([3.0,3.0]),
            'Q': np.array([3.0,0.0]),
            'R': np.array([3.0,2.0]),
            'S': np.array([10.0,3.0]),
            'T': np.array([10.0,1.0]),
            'U': np.array([11.0,1.0]),
            'V': np.array([4.0,3.0]),
        }
        self.doors = []
        self.Obstacles = ObstacleConstraintsGenerator(robot_dim=robot_dim, scale=scale)

    def generate_walls(self):
        points = np.array([
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
            start_pos = points[i][0] - self.offset
            end_pos = points[i][1] - self.offset
            self.add_wall(start_pos, end_pos)
        
        self.Obstacles.walls = np.array(self.Obstacles.walls)

    def add_wall(self, start_pos, end_pos):
        vec = end_pos - start_pos
        avg = (end_pos + start_pos)/2
        theta = np.arctan2(*vec)
    
        dim = np.array([WIDTH, np.linalg.norm(vec), HEIGHT])
        pos = [[avg[0], avg[1], theta]]
        self.Obstacles.walls.append({'x': pos[0][0], 'y': pos[0][1], 'theta': pos[0][2], 'width': dim[0], 'length': dim[1], 'height': dim[2]}) # Add new obstacle pos to list
        self.env.add_shapes(shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos)

    def generate_doors(self):
        # Add all door and door knobs to pos list and convert all lists to np arrays
        door_bathroom = Door(self.env, pos=self.points['U']-self.offset, is_open=True, theta=0)
        door_bathroom.draw_door()
        self.Obstacles.doors.append({'Bathroom door': {'x': door_bathroom.pos_door[0][0], 'y': door_bathroom.pos_door[0][1], 'theta': door_bathroom.pos_door[0][2], 
                                        'width': door_bathroom.dim_door[0], 'length': door_bathroom.dim_door[1], 'height': door_bathroom.dim_door[2]},
                                        'Bathroom doorknob': {'x': door_bathroom.pos_knob[0][0], 'y': door_bathroom.pos_knob[0][1], 'theta': door_bathroom.pos_knob[0][2], 
                                        'width': door_bathroom.dim_knob[0], 'length': door_bathroom.dim_knob[1], 'height': door_bathroom.dim_knob[2]}})

        door_outdoor = Door(self.env, pos=self.points['E']-self.offset, is_open=True, theta=np.pi)
        door_outdoor.draw_door()
        self.Obstacles.doors.append({'Out door': {'x': door_outdoor.pos_door[0][0], 'y': door_outdoor.pos_door[0][1], 'theta': door_outdoor.pos_door[0][2], 
                                        'width': door_outdoor.dim_door[0], 'length': door_outdoor.dim_door[1], 'height': door_outdoor.dim_door[2]},
                                        'Out doorknob': {'x': door_outdoor.pos_knob[0][0], 'y': door_outdoor.pos_knob[0][1], 'theta': door_outdoor.pos_knob[0][2], 
                                        'width': door_outdoor.dim_knob[0], 'length': door_outdoor.dim_knob[1], 'height': door_outdoor.dim_knob[2]}})

        door_bedroom1 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0, is_flipped=True)
        door_bedroom1.draw_door()
        self.Obstacles.doors.append({'Bedroom1 door': {'x': door_bedroom1.pos_door[0][0], 'y': door_bedroom1.pos_door[0][1], 'theta': door_bedroom1.pos_door[0][2], 
                                        'width': door_bedroom1.dim_door[0], 'length': door_bedroom1.dim_door[1], 'height': door_bedroom1.dim_door[2]},
                                        'Bedroom1 doorknob': {'x': door_bedroom1.pos_knob[0][0], 'y': door_bedroom1.pos_knob[0][1], 'theta': door_bedroom1.pos_knob[0][2], 
                                        'width': door_bedroom1.dim_knob[0], 'length': door_bedroom1.dim_knob[1], 'height': door_bedroom1.dim_knob[2]}})

        door_bedroom2 = Door(self.env, pos=self.points['R']-self.offset, is_open=True, theta=0.5*np.pi)
        door_bedroom2.draw_door()
        self.Obstacles.doors.append({'Bedroom2 door': {'x': door_bedroom2.pos_door[0][0], 'y': door_bedroom2.pos_door[0][1], 'theta': door_bedroom2.pos_door[0][2], 
                                        'width': door_bedroom2.dim_door[0], 'length': door_bedroom2.dim_door[1], 'height': door_bedroom2.dim_door[2]},
                                        'Bedroom2 doorknob': {'x': door_bedroom2.pos_knob[0][0], 'y': door_bedroom2.pos_knob[0][1], 'theta': door_bedroom2.pos_knob[0][2], 
                                        'width': door_bedroom2.dim_knob[0], 'length': door_bedroom2.dim_knob[1], 'height': door_bedroom2.dim_knob[2]}})

        door_kitchen = Door(self.env, pos=self.points['I']-self.offset, is_open=True, theta=-0.5*np.pi)
        door_kitchen.draw_door()
        self.Obstacles.doors.append({'Kitchen door': {'x': door_kitchen.pos_door[0][0], 'y': door_kitchen.pos_door[0][1], 'theta': door_kitchen.pos_door[0][2], 
                                        'width': door_kitchen.dim_door[0], 'length': door_kitchen.dim_door[1], 'height': door_kitchen.dim_door[2]},
                                        'Kitchen doorknob': {'x': door_kitchen.pos_knob[0][0], 'y': door_kitchen.pos_knob[0][1], 'theta': door_kitchen.pos_knob[0][2], 
                                        'width': door_kitchen.dim_knob[0], 'length': door_kitchen.dim_knob[1], 'height': door_kitchen.dim_knob[2]}})
        self.Obstacles.doors = np.array(self.Obstacles.doors)

