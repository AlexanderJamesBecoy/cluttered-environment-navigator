import gym
import numpy as np
from model import Model
from house import House
import warnings

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
        robots = [Model(),]

        env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )

        action = np.zeros(env.n())
        action[2] = 0.5

        ob = env.reset() # pos=...
        # generate_room(env)

        house = House(env)
        # start_pos = np.array([0.0, 1.0])
        # end_pos = np.array([1.0, 1.0])
        # house.add_wall(start_pos, end_pos)
        house.generate_walls()

        print(f"Length: {len(action)}")
        print(f"Initial observation : {ob}")
        history = []

        for i in range(1000):
            # if (int(i / 100)) % 2 == 0:
            #     action[-1] = -0.01
            #     action[-2] = -0.01
            # else:
            #     action[-1] = 0.01
            #     action[-2] = 0.01
            ob, _, _, _ = env.step(action)
            history.append(ob)
        
        env.close()