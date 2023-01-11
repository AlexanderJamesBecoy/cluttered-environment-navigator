import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model import Model
from house import House
from planner import Planner
import warnings

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
        env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )
        house = House(env, robot_dim=robot_dim, scale=R_SCALE)
        planner = Planner(robot=robots[0], house=house)

        # History
        history = []

        # Generate environment
        # start_pos = robots[0].set_initial_pos(3.0,-2.0)
        start_pos, open_doors = planner.plan_motion()
        init_joints = robots[0].set_initial_pos(start_pos)
        ob = env.reset(pos=init_joints)
        house.generate_walls()
        house.generate_doors(open_doors)
        house.generate_furniture()
        planner.plot_plan_2d()

        # Follow a path set by waypoints
        waypoints = planner.generate_waypoints()
        robots[0].follow_path(env=env, house=house, waypoints=waypoints)

        env.close()
