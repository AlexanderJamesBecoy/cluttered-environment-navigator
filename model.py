import gym
import numpy as np
from gym_envs_urdf.urdfenvs.robots.albert import AlbertRobot
from gym_envs_urdf.urdfenvs.robots.generic_urdf import GenericUrdfReacher

def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        AlbertRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(9)
    action[0] = 5.0
    ob = env.reset(
        pos=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history

def run_mobile_reacher(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="mobilePandaWithGripper.urdf", mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    action[2] = 0.5 
    # action[2] = 0.5
    # action[5] = -0.0
    # action[-1] = 3.5
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    for i in range(n_steps):
        if (int(i / 100)) % 2 == 0:
            action[-1] = -0.01
            action[-2] = -0.01
        else:
            action[-1] = 0.01
            action[-2] = 0.01
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history
