import gym
import numpy as np
from model import Model
from house import House
import warnings
from controller import MPController

R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check

#Dimension of robot base, found in mobilePandaWithGripper.urdf
R_RADIUS = 0.2
R_HEIGHT = 0.3


def generate_room(env):
    # # Dimensions
    dim_wall_3 = np.array([0.3,3.0,0.5]) # three-meter long wall
    dim_wall_2 = np.array([0.3,2.0,0.5]) # two-meter long wall
    dim_wall_1 = np.array([0.3,1.0,0.5]) # one-meter long wall

    # # Poses
    # poses_wall_3 = [
    #     [-3.0, -2.5, 0.0],         # Bedroom B
    #     [-1.5, -4.0, 0.5*np.pi],   # Bedroom B
    #     [0.0, -2.5, 0.0],          # Bedroom B/Liv.Room
    #     [1.5, -4.0, 0.5*np.pi],    # Liv.Room
    #     [3.0, -2.5, 0.0],          # Liv.Room
    #     [3.0, 0.5, 0.0],           # Din.Room
    #     [1.5, 4.0, 0.5*np.pi],      # Kitchen
    #     [0.0, 2.5, 0.0],      # Kitchen/Bedroom A
    #     [-1.5, 4.0, 0.5*np.pi],     # Bedroom A
    #     [-3.0, 2.5, 0.0],           # Bedroom A
    #     [-2.5, -1.0, 0.5*np.pi],    # Bedroom B/Bathroom
    #     [-2.5, 1.0, 0.5*np.pi],     # Bedroom A/Bathroom
    # ]
    # poses_wall_2 = [
    #     [3.0, 3.0, 0.0],            # Kitchen
    #     [-4.0, 0.0, 0.0],           # Bathroom
    # ]

    # Poses
    # poses_wall_3 = [
    #     []
    # ]

    # env.add_shapes(shape_type="GEOM_BOX", dim=dim_wall_3, mass=0, poses_2d=poses_wall_3)
    # env.add_shapes(shape_type="GEOM_BOX", dim=dim_wall_2, mass=0, poses_2d=poses_wall_2)


if __name__ == "__main__":

    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)

        robot_dim = np.array([R_HEIGHT, R_RADIUS])
        robots = [Model(dim=robot_dim),]
        robots[0]._urdf.center
        env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )

        action = np.zeros(env.n())

        ob = env.reset() # pos=...
        house = House(env, robot_dim=robot_dim, scale=R_SCALE)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }

        house.generate_walls()
        house.generate_doors(is_open)

        # Generate obstacle constraints
        left, right, low, up = house.Obstacles.generateConstraintsCylinder()
        print(left.shape, right.shape, low.shape, up.shape)

        print(f"Length: {len(action)}")
        print(f"Initial observation : {ob}")
        history = []

        # Target position of the robot
        goal = np.array([0, -3, 3.14, 3.14/2, 3.14/2, 3.14/2, 0])
        MPC = MPController(robots[0])

        while(1):
            ob, _, _, _ = env.step(action)
            state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
            actionMPC = MPC.FHOCP(state0, goal)
            action = np.zeros(env.n())
            for i, j in enumerate(robots[0]._dofs):
                action[j] = actionMPC[i]

            # history.append(ob)

        env.close()