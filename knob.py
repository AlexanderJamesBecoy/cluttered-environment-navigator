import gym
import numpy as np
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

class Knob:
# This class simulates a simple door knob. It only acts a static entity and is used as a goal for our mobile manipulator.

    def __init__(self, env, pos_3d, radius=0.1):
    # Declare private variables from the given arguments.
        self.env = env
        self.pos = pos_3d
        self.radius = radius

    def draw_knob(self): # TODO -> into goal because add_shapes create obstacles
    # Draw the knob
        dim = np.array([self.radius])
        pos = [[self.pos[0], self.pos[1], 0]]
        self.env.add_shapes(shape_type="GEOM_SPHERE", dim=dim, mass=0, poses_2d=pos, place_height=self.pos[2])

    def get_pos(self):
    # Return the 3D position of the door knob.
        return self.pos
