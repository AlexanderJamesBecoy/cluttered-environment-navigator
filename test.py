# import gym
# import numpy as np
# from model import Model
# from house import House
# import warnings
# from MPC import MPController

# R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check

# #Dimension of robot base, found in mobilePandaWithGripper.urdf
# R_RADIUS = 0.2
# R_HEIGHT = 0.3


# def generate_room(env):
#     # # Dimensions
#     dim_wall_3 = np.array([0.3,3.0,0.5]) # three-meter long wall
#     dim_wall_2 = np.array([0.3,2.0,0.5]) # two-meter long wall
#     dim_wall_1 = np.array([0.3,1.0,0.5]) # one-meter long wall

#     # # Poses
#     # poses_wall_3 = [
#     #     [-3.0, -2.5, 0.0],         # Bedroom B
#     #     [-1.5, -4.0, 0.5*np.pi],   # Bedroom B
#     #     [0.0, -2.5, 0.0],          # Bedroom B/Liv.Room
#     #     [1.5, -4.0, 0.5*np.pi],    # Liv.Room
#     #     [3.0, -2.5, 0.0],          # Liv.Room
#     #     [3.0, 0.5, 0.0],           # Din.Room
#     #     [1.5, 4.0, 0.5*np.pi],      # Kitchen
#     #     [0.0, 2.5, 0.0],      # Kitchen/Bedroom A
#     #     [-1.5, 4.0, 0.5*np.pi],     # Bedroom A
#     #     [-3.0, 2.5, 0.0],           # Bedroom A
#     #     [-2.5, -1.0, 0.5*np.pi],    # Bedroom B/Bathroom
#     #     [-2.5, 1.0, 0.5*np.pi],     # Bedroom A/Bathroom
#     # ]
#     # poses_wall_2 = [
#     #     [3.0, 3.0, 0.0],            # Kitchen
#     #     [-4.0, 0.0, 0.0],           # Bathroom
#     # ]

#     # Poses
#     # poses_wall_3 = [
#     #     []
#     # ]

#     # env.add_shapes(shape_type="GEOM_BOX", dim=dim_wall_3, mass=0, poses_2d=poses_wall_3)
#     # env.add_shapes(shape_type="GEOM_BOX", dim=dim_wall_2, mass=0, poses_2d=poses_wall_2)


# if __name__ == "__main__":

#     show_warnings = False
#     warning_flag = "default" if show_warnings else "ignore"
#     with warnings.catch_warnings():
#         warnings.filterwarnings(warning_flag)

#         robot_dim = np.array([R_HEIGHT, R_RADIUS])
#         robots = [Model(dim=robot_dim),]
#         robots[0]._urdf.center
#         env = gym.make(
#             "urdf-env-v0",
#             dt=0.01, robots=robots, render=True
#         )

#         action = np.zeros(env.n())

#         ob = env.reset() # pos=...
#         house = House(env, robot_dim=robot_dim, scale=R_SCALE)
#         is_open = {
#             'bathroom':         True,
#             'outdoor':          True,
#             'top_bedroom':      True,
#             'bottom_bedroom':   True,
#             'kitchen':          True,
#         }

#         house.generate_walls()
#         house.generate_doors(is_open)

#         print(f"Length: {len(action)}")
#         print(f"Initial observation : {ob}")
#         history = []

#         # Target position of the robot
#         goal = np.array([0, -3, 3.14, 3.14/2, 3.14/2, 3.14/2, 0])
#         MPC = MPController(robots[0])

#         while(1):
#             ob, _, _, _ = env.step(action)
#             state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
#             actionMPC = MPC.solve_MPC(state0, goal)
#             action = np.zeros(env.n())
#             for i, j in enumerate(robots[0]._dofs):
#                 action[j] = actionMPC[i]

#             # history.append(ob)

#         env.close()


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
        # plot_2d(lines, boxes)

        # print(f"Length: {len(action)}")
        # print(f"Initial observation : {ob}")
        # history = []

        # Target position of the robot  
        waypoints = np.array([[0, -2], [2, -2], [2, 0], [0, 0], [0, 10], [10, 10], [-10, -10]])

        # Follow a path set by waypoints
        # robots[0].follow_path(env=env, house=house, waypoints=waypoints)
        MPC = MPController(robots[0])
        goal = np.array([-1, 0, 0, 3.14/2, 3.14/2, 3.14/2, 0])
        while(1):
            ob, _, _, _ = env.step(action)
            state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
            actionMPC = MPC.solve_MPC(state0, goal)
            action = np.zeros(env.n())
            for i, j in enumerate(robots[0]._dofs):
                action[j] = actionMPC[i]
        env.close()
