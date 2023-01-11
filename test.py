import gym
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model import Model
from house import House
from planner import Planner
import warnings
from MPC import MPController
import time
from free_space import FreeSpace

TEST_MODE = True # Boolean to initialize test mode to test the MPC
R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check

#Dimension of robot base, found in mobilePandaWithGripper.urdf
R_RADIUS = 0.2
R_HEIGHT = 0.3

if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)

        # Main init
        robot_dim = np.array([R_HEIGHT, R_RADIUS])
        robots = [Model(dim=robot_dim),]
        robots[0]._urdf.center
        env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=True)
        house = House(env, robot_dim=robot_dim, scale=R_SCALE, test_mode=TEST_MODE)
        env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )
        house = House(env, robot_dim=robot_dim, scale=R_SCALE)


        # Generate environment
        start_pos = robots[0].set_initial_pos([3.0, -2.0])
        ob = env.reset(pos=start_pos)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }
        house.generate_walls()
        # house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=TEST_MODE)
        no_rooms = planner.plan_motion()
        house.generate_doors()
        # house.generate_furniture()

        # History
        history = []

        for room in range(no_rooms):
            # Generate environment
            route, open = planner.generate_waypoints(room)
            init_joints = robots[0].set_initial_pos(route[0])
            ob = env.reset(pos=init_joints)
            house.draw_walls()
            # house.draw_doors(open)
            house.draw_furniture()
            planner.plot_plan_2d(route)

            # Follow a path set by waypoints   z
            MPC = MPController(robots[0])
            goal = np.array([3, 3, 0, 0, 0, 0, 0])
            action = np.zeros(env.n())
            k = 0
            while(1):
                ob, _, _, _ = env.step(action)
                _, b, A, vertices = house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
                print("A: \n{}\nb: \n{}\n".format(A, b))
                zero_col = np.zeros((b.size, 1))
                A = np.hstack((A, zero_col))
                state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
                actionMPC = MPC.solve_MPC(state0, goal, A, b)
                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]
        # Target position of the robot  
        waypoints = np.array([[0, -2], [2, -2], [2, 0], [0, 0], [0, 10], [10, 10], [-10, -10]])

        # Follow a path set by waypoints
        # robots[0].follow_path(env=env, house=house, waypoints=waypoints)
        MPC = MPController(robots[0])
        goal = np.array([3, -5, 0, 0, 0, 0, 0])
        action = np.zeros(env.n())
        k = 0


        while(1):
            ob, _, _, _ = env.step(action)
            _, b, A, vertices = house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
            zero_col = np.zeros((b.size, 1))
            A = np.hstack((A, zero_col))
            state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
            # Cfree = FreeSpace(vertices, state0)
            # A, b = Cfree.update_free_space(state0)

            actionMPC = MPC.solve_MPC(state0, goal, A, b)
            action = np.zeros(env.n())
            for i, j in enumerate(robots[0]._dofs):
                action[j] = actionMPC[i]

                if (k%50 == 0):
                    house.Obstacles.display()
                k += 1
        env.close()