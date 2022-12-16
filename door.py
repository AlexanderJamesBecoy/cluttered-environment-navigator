import gym
import numpy as np

class Door:
    
    def __init__(self, env, pos, theta, is_flipped=False, is_open=False):
        self.env = env
        self.pos = pos
        self.theta = theta
        self.flipped = 1
        self.open = 0
        self.dim_door = np.array([1.0, 0.1, 2.0])
        self.dim_knob = np.array([0.2, 0.3, 0.2]) # TODO -> into goal object
        self.pos_door = []
        self.pos_knob = []

        if (is_flipped):
            self.flipped = -1
        if (is_open):
            self.open = 0.5*np.pi
    
    def draw_door(self):
        offset_x = 0.5*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y = 0.5*np.sin(self.theta+self.open*self.flipped)*self.flipped

        offset_x_knob = 0.3*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y_knob = 0.3*np.sin(self.theta+self.open*self.flipped)*self.flipped

        self.pos_door = [[self.pos[0]+offset_x, self.pos[1]+offset_y, self.theta+self.open*self.flipped]]
        self.pos_knob = [[self.pos[0]+offset_x+offset_x_knob, self.pos[1]+offset_y+offset_y_knob, self.theta+self.open*self.flipped]]

        self.env.add_shapes(shape_type="GEOM_BOX", dim=self.dim_door, mass=0, poses_2d=self.pos_door)
        self.env.add_shapes(shape_type="GEOM_BOX",dim=self.dim_knob, mass=0, poses_2d=self.pos_knob, place_height=1.0)

        