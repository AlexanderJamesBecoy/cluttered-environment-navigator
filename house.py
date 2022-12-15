import gym
import numpy as np

class House:
    def __init__(self, env):
        self.env = env
        self.offset = np.array([7.0, 3.5])

    def generate_walls(self):
        points = np.array([
            [[0.0,0.0], [12.0,0.0]],
            [[0.0,0.0], [0.0,7.0]],
            [[0.0,7.0], [7.0,7.0]],
            [[8.0,7.0], [11.0,7.0]],
            [[11.0,7.0], [11.0,3.0]],
            [[12.0,0.0], [12.0,3.0]],
            [[8.0,3.0], [12.0,3.0]],
            [[5.0,7.0], [5.0,3.0]],
            [[6.0,7.0], [6.0,3.0]],
            [[5.0,3.0], [6.0,3.0]],
            [[8.0,7.0], [8.0,4.0]],
            [[0.0,3.0], [4.0,3.0]],
            [[3.0,0.0], [3.0,2.0]],
            [[10.0,3.0], [10.0,1.0]],
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