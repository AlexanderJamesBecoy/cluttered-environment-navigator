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
from free_space import FreeSpace
import time

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
        house.generate_walls()
        # house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=TEST_MODE)
        no_rooms = planner.plan_motion()

        # History
        history = []

        for room in range(no_rooms):
            # Generate environment
            route, open = planner.generate_waypoints(room)
            init_joints = robots[0].set_initial_pos(route[0])
            start_pos = robots[0].set_initial_pos([-1, -1.])
            ob = env.reset(pos=start_pos)
            house.draw_walls()
            # house.draw_doors(open)
            house.draw_furniture()
            planner.plot_plan_2d(route)

            # Follow a path set by waypoints   z
            MPC = MPController(robots[0])
            goal = np.array([0, 1.5, 0, 0, 0, 0, 0])
            action = np.zeros(env.n())
            k = 0
            vertices = house.Obstacles.getVertices()
            while(1):
                ob, _, _, _ = env.step(action)
                # _, b, A, vertices = house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
                # print("A: \n{}\nb: \n{}\n".format(A, b))
                # zero_col = np.zeros((b.size, 1))
                # A = np.hstack((A, zero_col))
                state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]

                # floor = [[4, 4, 0], [4, -4, 0], [-4, -4, 0], [4, -4, 0], [4, 4, -0.1], [4, -4, -0.1], [-4, -4, -0.1], [4, -4, -0.1]]
                # ceiling = [[4, 4, 1.5], [4, -4, 1.5], [-4, -4, 1.5], [4, -4, 1.5], [4, 4, 1.6], [4, -4, 1.6], [-4, -4, 1.6], [4, -4, 1.6]]
                # vertices = vertices.tolist()
                # vertices.append(floor)
                # vertices.append(ceiling)
                # obstacles = []
                # for vertex in vertices:
                #     obstacles.append(np.array(vertex))
                if (k%10 == 0):

                    floor = np.array([[4, 4, 0], [-4, 4, 0], [-4, -4, 0], [4, -4, 0], [4, 4, -0.1], [-4, 4, -0.1], [-4, -4, -0.1], [4, -4, -0.1]])
                    ceiling = np.array([[4, 4, 1.5], [-4, 4, 1.5], [-4, -4, 1.5], [4, -4, 1.5], [4, 4, 1.6], [-4, 4, 1.6], [-4, -4, 1.6], [4, -4, 1.6]])
                    Hlow = np.array([[-3.75, -3.7, 0], [3.75, -3.7, 0], [3.75, -3.8, 0], [-3.75, -3.8, 0], [-3.75, -3.7, 1.5], [3.75, -3.7, 1.5], [3.75, -3.8, 1.5], [-3.75, -3.8, 1.5]])
                    Hhigh = np.array([[-3.75, 3.7, 0], [3.75, 3.7, 0], [3.75, 3.8, 0], [-3.75, 3.8, 0], [-3.75, 3.7, 1.5], [3.75, 3.7, 1.5], [3.75, 3.8, 1.5], [-3.75, 3.8, 1.5]])
                    Vleft = np.array([[-3.8, 3.75, 0], [-3.7, 3.75, 0], [-3.7, -3.75, 0], [-3.8, -3.75, 0], [-3.8, 3.75, 1.5], [-3.7, 3.75, 1.5], [-3.7, -3.75, 1.5], [-3.8, -3.75, 1.5]])
                    Vright = np.array([[3.8, 3.75, 0], [3.7, 3.75, 0], [3.7, -3.75, 0], [3.8, -3.75, 0], [3.8, 3.75, 1.5], [3.7, 3.75, 1.5], [3.7, -3.75, 1.5], [3.8, -3.75, 1.5]])
                    block = np.array([[0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 1.5], [0.5, 0.5, 1.5], [-0.5, 0.5, 1.5], [-0.5, -0.5, 1.5]])
                    obstacles = [floor, ceiling, Hlow, Hhigh, Vleft, Vright, block]
                    p0 = [state0[0], state0[1], 0.3 + 0.1]
                    print(p0)
                    Cfree = FreeSpace(obstacles, p0)
                    A, b = Cfree.update_free_space(p0)
                k += 1

                start_time = time.time()
                actionMPC = MPC.solve_MPC(state0, goal, A, b)
                end_time = time.time()
                print("MPC computation time: ", end_time - start_time)

                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]

                # if (k%50 == 0):
                #     house.Obstacles.display()
                # k += 1
        env.close()