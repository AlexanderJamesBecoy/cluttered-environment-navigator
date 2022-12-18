import numpy as np
# from gym_envs_urdf.urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
import os
import sys
sys.path.insert(1, '../gym_envs_urdf')

class Model(HolonomicRobot):
    def __init__(self, urdf="../urdfenvs/robots/generic_urdf/mobile_panda/mobilePandaWithGripper.urdf", mode="vel"):
        self._urdf = urdf
        self.dofs = [0, 1, 2, 4, 6, 8, 9]   # 0/0 - x-direction
                                            # 1/1 - y-direction
                                            # 2/2 - yaw
                                            # 3/4 - arm joint 1
                                            # 4/6 - arm joint 2
                                            # 5/8 - arm joint 3
                                            # 6/9 - arm joint 4
        #self.states = {'x': 0, 'y': 0, 'theta': 0, 'v_x': 0, 'v_y': 0, 'v_r': 0}
        #search for urdf in package if not found in cwd
        if not os.path.exists(urdf):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            urdf = None
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file == self._urdf:
                        urdf = os.path.join(root, file)
            if urdf is None:
                raise Exception(f"the request urdf {self._urdf} can not be found")
            self._urdf = urdf

        super().__init__(-1, self._urdf, mode=mode)

    def act(self, joints):
        return self.dofs[joints]

    def set_joint_names(self):
        # TODO Replace it with a automated extraction
        self._joint_names = [joint.name for joint in self._urdf_robot._actuated_joints]

    def set_acceleration_limits(self):
        acc_limit = np.array(
            # [1.0, 1.0, 15.0, 7.5, 12.5, 20.0, 20.0]
            [1.0, 1.0, 15.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0]
        )
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]

    def check_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.n():
            center_position = (self._limit_pos_j[0] + self._limit_pos_j[1])/2
            pos = center_position
        if not isinstance(vel, np.ndarray) or not vel.size == self.n():
            vel = np.zeros(self.n())
        return pos, vel

    def move_to_waypoint(self, waypoint: np.ndarray, obs: dict) -> None:
        """
            Move the robot to the target waypoint, in Euclidean direction (i.e. straight line)
            Waypoint should be a 2-elements vector containing the x and y coordinates of the point
        """
        # Get current x and y positions
        x = obs['robot_0']['joint_state']['position'][0]
        y = obs['robot_0']['joint_state']['position'][1]

        # Approximate to zero
        if np.abs(x) < 1e-04:
            x = 0.0
        
        if np.abs(y) < 1e-04:
            y = 0.0
        
        vel = np.zeros(self._n) # action

        targetVector = np.array([waypoint[0] - x, waypoint[1] - y])

        # Prevent dividing by zero
        if targetVector[0] == 0.0:
            vel[:2] = np.array((waypoint - np.array([x ,y]))/np.abs(np.array([waypoint[1] - y])))
        elif targetVector[1] == 0.0:
            vel[:2] = np.array((waypoint - np.array([x ,y]))/np.abs(np.array([waypoint[0] - x])))
        else:
            vel[:2] = np.array((waypoint - np.array([x ,y]))/np.abs(np.array([waypoint[0] - x, waypoint[1] - y])))
        self.update_state()

        return vel, np.allclose(np.array([x, y]), waypoint, rtol=1e-03, atol=1e-03)
