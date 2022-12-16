import gym
from door import Door
import numpy as np

class House:
    def __init__(self, env):
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

    def add_wall(self, start_pos, end_pos):
        vec = end_pos - start_pos
        avg = (end_pos + start_pos)/2
        theta = np.arctan2(*vec)
    
        dim = np.array([0.3, np.linalg.norm(vec), 0.5])
        pos = [[avg[0], avg[1], theta]]
        self.env.add_shapes(shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos)

    def generate_doors(self):
        door_bathroom = Door(self.env, pos=self.points['U']-self.offset, is_open=True, theta=0)
        door_bathroom.draw_door()
        door_outdoor = Door(self.env, pos=self.points['E']-self.offset, is_open=True, theta=np.pi)
        door_outdoor.draw_door()
        door_bedroom1 = Door(self.env, pos=self.points['P']-self.offset, is_open=True, theta=0, is_flipped=True)
        door_bedroom1.draw_door()
        door_bedroom2 = Door(self.env, pos=self.points['R']-self.offset, is_open=True, theta=0.5*np.pi)
        door_bedroom2.draw_door()
        door_kitchen = Door(self.env, pos=self.points['I']-self.offset, is_open=True, theta=-0.5*np.pi)
        door_kitchen.draw_door()


