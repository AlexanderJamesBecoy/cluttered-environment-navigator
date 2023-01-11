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

R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check

#Dimension of robot base, found in mobilePandaWithGripper.urdf
R_RADIUS = 0.2
R_HEIGHT = 0.3

# Matplotlib
def plot_2d(lines, boxes):
    # Generate 2D plot of house
    fig, ax = plt.subplots()
    for line in lines:
        x = np.array(line['coord'])[:,0]
        y = np.array(line['coord'])[:,1]
        if line['type'] == 'wall':
            color = 'k'
        else:
            color = 'r'
        ax.plot(x,y, color, linewidth=2)
    for box in boxes:
        ax.add_patch(
            Rectangle((box['x'],box['y']),box['w'],box['h'],
            facecolor='blue',
            fill=True,
        ))
    plt.show()

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


        # Generate environment
        start_pos = robots[0].set_initial_pos(3.0,-2.0)
        ob = env.reset(pos=start_pos)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }
        house.generate_walls()
        house.generate_doors(is_open)
        house.generate_furniture()

        lines, boxes = house.generate_plot_obstacles()
        plot_2d(lines, boxes)

        # print(f"Length: {len(action)}")
        # print(f"Initial observation : {ob}")
        # history = []

        # Target position of the robot  
        waypoints = np.array([[0, -2], [2, -2], [2, 0], [0, 0], [0, 10], [10, 10], [-10, -10]])

        # Follow a path set by waypoints
        robots[0].follow_path(env=env, house=house, waypoints=waypoints)

        env.close()
