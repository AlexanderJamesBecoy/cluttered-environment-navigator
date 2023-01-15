import gym
import numpy as np
from model import Model
from house import House
from planner import Planner
import warnings
import time
import matplotlib.pyplot as plt
import os

TEST_MODE = False # Boolean to initialize test mode to test the MPC
DEBUG_MODE = False # Boolean to display the states.
R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check

#Dimension of robot base, found in mobilePandaWithGripper.urdf
R_RADIUS = 0.2
R_HEIGHT = 0.3

max_time = []
waypoints_achieveds = []
nos_waypoints = []
time_history = []
rrt_time = 0
step_size = 0.25
start = 'top_bedroom'
end = 'bathroom'
pos = [0.,0.]
dist_travelled = []
rrt_times = []
rrt_dists = []

if __name__ == "__main__":

    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)

        # Main init
        robot_dim = np.array([R_HEIGHT, R_RADIUS])
        robots = [Model(dim=robot_dim),]
        robots[0]._urdf.center
        env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=False)
        house = House(env, robot_dim=robot_dim, scale=R_SCALE, test_mode=TEST_MODE)
        house.generate_walls()
        house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=TEST_MODE, debug_mode=DEBUG_MODE)
        # no_rooms = planner.plan_motion(start=[-1.5,-4.5], end=[1.5,4.5], step_size=0.2, max_iter=5000)
        no_rooms = 0
        for i in range(100):
            print(i)
            start_time = time.time()
            no_rooms, rrt_dist, pos[0], pos[1] = planner.plan_motion(start=start, end=end, step_size=step_size, max_iter=10000)
            rrt_time = time.time() - start_time
            rrt_times.append(rrt_time)
            rrt_dists.append(rrt_dist)

        with open('rrt_data.txt', 'w') as f:
            f.write("Time: \n")
            for rrt_time in rrt_times:
                f.write(f"{rrt_time}\n")
            f.write("\n Cost: \n")
            for rrt_dist in rrt_dists:
                f.write(f"{rrt_dist}\n")

        # History
        history = []

        for i in range(no_rooms):
            # Generate environment
            route, open = planner.generate_waypoints(i)
            init_joints = robots[0].set_initial_pos(route[0])
            ob = env.reset(pos=init_joints)
            house.draw_walls()
            house.draw_doors(open)
            house.draw_furniture()
            planner.plot_plan_2d(i)
            start_time = time.time()

            # Follow a path set by waypoints   z
            waypoints_achieved, no_waypoints,robot_dist = robots[0].follow_path(env=env, house=house, waypoints=route)
            time_diff = time.time() - start_time
            time_history.append(time_diff)
            waypoints_achieveds.append(waypoints_achieved)
            nos_waypoints.append(no_waypoints)
            dist_travelled.append(robot_dist)

        env.close()

    # Plot
    # print(f'{start} to {end}')
    # print(f"RRT execution time: {rrt_time}")
    # print(f"RRT step size: {step_size}")
    # print(f"Total number of waypoints in room: {np.sum(np.array(nos_waypoints))}")
    # print(f"Number of waypoints achieved in room: {np.sum(np.array(waypoints_achieveds))}")
    # print(f"Completion time in room: {np.sum(np.array(time_history))}")
    # dist = np.linalg.norm(np.array(pos[1]) - np.array(pos[0]))
    # print(f"Distance between starting point and end point: {dist}")
    # print(f"Distance calculated: {rrt_dist}")
    # print(f"Distance travlled: {np.sum(np.array(dist_travelled))}")
    # # print(f"RRT step size: {step_size}")
    # x_rooms = range(no_rooms)
    # plt.subplot(2,1,1)
    # plt.plot(x_rooms, nos_waypoints, color='blue', label='total waypoints')
    # plt.plot(x_rooms, waypoints_achieveds, color='red', label='waypoints achieved')
    # plt.xlabel('Room')
    # plt.ylabel('Waypoints')
    # plt.subplot(2,1,2)
    # plt.bar(x_rooms, time_history, color='blue', label='Completion time')
    # plt.xlabel('Room')
    # plt.ylabel('Time [s]')
    # plt.suptitle(f'{start} to {end}')
    # plt.show()
