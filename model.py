import numpy as np
from gym_envs_urdf.urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
# from urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
import os
import sys
# sys.path.insert(1, '../gym_envs_urdf')

class Model(HolonomicRobot):
    def __init__(self, urdf="mobilePandaWithGripper.urdf", mode="vel"):
        self._urdf = urdf
        self.dofs = [0, 1, 2, 4, 6, 8, 9]   # 0/0 - x-direction
                                            # 1/1 - y-direction
                                            # 2/2 - yaw
                                            # 3/4 - arm joint 1
                                            # 4/6 - arm joint 2
                                            # 5/8 - arm joint 3
                                            # 6/9 - arm joint 4

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