import gym
import numpy as np
from knob import Knob
from MotionPlanningEnv.cubeObstacle import CubeObstacle

class Door:
#  This class contains the useful information describing a door. It also contains the goal object `Knob`
# for which our mobile manipulator tries to reach.
    
    def __init__(self, env, pos, theta, is_flipped=False, is_open=False):
    # Initialize an object of this class. It requires the pointer to the gym environment `env`, the orientation of the door `theta`.
    # Booleans describing whether the door is mirrored `is_flipped`, and whether the door is open `is_open` are set to False by default.

        self.env = env
        self.pos = pos
        self.theta = theta
        self.flipped = 1    # No mirroring of poses.
        self.open = 0       # No additive angle.

        if (is_flipped):    # If the door is mirrored, -1 will mirror the required poses.
            self.flipped = -1
        if (is_open):       # If the door is open, an additive angle is added to rotate the door by additional 90 deg.
            self.open = 0.5*np.pi

        knobs = []  # List of door knob objects.
    
    def draw_door(self):
    # This function draws the door into gym `env`. No further passing of arguments required.

        # Obtain the dimension of the door and the doorknob, respectively.
        dim_door = np.array([1.0, 0.05, 2.0])
        dim_knob = np.array([0.1, 0.2, 0.1]) # TODO -> into goal object
            
        # Poses of 2D offset away from the hinge to draw the door. This is due to `env` drawing the shapes centered.
        offset_x = 0.5*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y = 0.5*np.sin(self.theta+self.open*self.flipped)*self.flipped

        # Poses of 2D offset away from the center of the door to draw the doorknob. This is due to `env` drawing the shapes centered.
        offset_x_knob = 0.3*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y_knob = 0.3*np.sin(self.theta+self.open*self.flipped)*self.flipped

        # Absolute 2D poses describing the centered positions of the door and the doorknob, respectively.
        pos_door = [[self.pos[0]+offset_x, self.pos[1]+offset_y, self.theta+self.open*self.flipped]]
        pos_knob = [[self.pos[0]+offset_x+offset_x_knob, self.pos[1]+offset_y+offset_y_knob, self.theta+self.open*self.flipped]]

        # Draw the door and doorknob.
        self.env.add_shapes(shape_type="GEOM_BOX", dim=dim_door, mass=0, poses_2d=pos_door)
        self.env.add_shapes(shape_type="GEOM_BOX",dim=dim_knob, mass=0, poses_2d=pos_knob, place_height=1.0)

        # Create door knob objects
        knobs_offset = np.array([0.15*np.sin(self.theta+self.open*self.flipped), 0.15*np.cos(self.theta+self.open*self.flipped), 0])
        print(knobs_offset)
        print(np.array(pos_knob)[0][0:2])
        pos_knob_1 = np.hstack((np.array(pos_knob)[0][0:2], np.array([1.0]))) - knobs_offset
        print(pos_knob_1)
        pos_knob_2 = np.hstack((np.array(pos_knob)[0][0:2], np.array([1.0]))) + knobs_offset
        knob_1 = Knob(self.env, pos_knob_1)
        knob_2 = Knob(self.env, pos_knob_2)
        knob_1.draw_knob()
        knob_2.draw_knob()