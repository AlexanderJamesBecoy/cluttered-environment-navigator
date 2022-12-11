import gym
import numpy as np
from model import Model
import warnings

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

        ob = env.reset()
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