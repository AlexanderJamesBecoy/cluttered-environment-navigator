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

TEST_MODE = False # Boolean to initialize test mode to test the MPC
DEBUG_MODE = True # Boolean to display the states.
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

        # Generate environment
        start_pos = robots[0].set_initial_pos([3.0,-2.0])
        ob = env.reset(pos=start_pos)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }
        house.generate_walls()
        house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=TEST_MODE, debug_mode=DEBUG_MODE)
        _ = planner.plan_motion(start=[-1.5,-4.5], end=[1.5,4.5], step_size=1.0, max_iter=5000)

        # History
        history = []

        for i in range(1):
            # Generate environment
            route, open = planner.generate_waypoints()
            init_joints = robots[0].set_initial_pos(route[0])
            ob = env.reset(pos=init_joints)
            house.draw_walls()
            house.draw_doors(open)
            house.draw_furniture()
            planner.plot_plan_2d(i)

            # Follow a path set by waypoints   z
            robots[0].follow_path(env=env, house=house, waypoints=route)

        env.close()
