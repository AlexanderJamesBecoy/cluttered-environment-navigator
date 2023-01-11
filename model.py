import numpy as np
from gym_envs_urdf.urdfenvs.urdf_common.holonomic_robot import HolonomicRobot
import os

class Model(HolonomicRobot):
    def __init__(self, dim, urdf="mobilePandaWithGripper.urdf", mode="vel"):
        self._urdf = urdf
        self._dim = dim
        self._dofs = [0, 1, 2, 4, 6, 8, 9]   # 0/0 - x-direction
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

    def set_initial_pos(self, pos):
        return np.array([pos[0],pos[1],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype=np.float32)

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

    def set_waypoint_action(self, house, waypoint: np.ndarray, obs: dict, ztol: float, rtol: float, atol: float) -> None:
        """
            Set the action necessary to reach the target waypoint.
            
            Returns:
                1x7 velocity vector and variable to denote whether the point has been reached or not
        """
        # Get current x and y positions
        x = obs['joint_state']['position'][0]
        y = obs['joint_state']['position'][1]
        rob, con = house.Obstacles.generateConstraintsCylinder([x, y], 2)
        print("Rob: {}\nConstraints: {}\n".format(rob, con))
        vel = np.zeros(self._n) # action
        targetVector = np.array([waypoint[0] - x, waypoint[1] - y])

        vel[:2] = np.array(targetVector/np.linalg.norm(targetVector))

        # Check if current robot position is within tolerated range
        return vel, np.allclose(np.array([x, y]), waypoint, rtol=rtol, atol=atol)
    
    def follow_path(self, env, house, waypoints: np.ndarray, iter: int=1000, ztol=1e-03, rtol=1e-02, atol=1e-02) -> None:
        """
            Iterate points over waypoints and move the robot to each one sequentially.
            Maximum iteration set by 'iter'.

            Program exits when either maximum iteration has been reached or all waypoints visited.
        """

        for point in waypoints:
            done = False
            i = 0
            while (not done and i < iter):
                action, done = self.set_waypoint_action(house, point, self.state, ztol=ztol, rtol=rtol, atol=atol)
                env.step(action)
                self.update_state()
                i += 1

