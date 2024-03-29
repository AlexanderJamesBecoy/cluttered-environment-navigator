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
from arm_MPC import ArmMPController
from free_space import FreeSpace
import time
from drawing import draw_region

TEST_MODE = True # Boolean to initialize test mode to test the MPC
R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check
METHOD = ''

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
        planner = Planner(house=house, test_mode=TEST_MODE, doors_exist=False)
        no_rooms = planner.plan_motion(start=[-2, 0], end=[2, 0])

        # History
        history = []

        for room in range(no_rooms):
            # Generate environment
            route = planner.generate_waypoints(room)
            init_joints = robots[0].set_initial_pos(route[0][0])
            start_pos = robots[0].set_initial_pos([-2, 0])
            ob = env.reset(pos=start_pos)
            house.draw_walls()
            # house.draw_doors(open)
            house.draw_furniture()
            planner.plot_plan_2d(route)

            # Follow a path set by waypoints z
            MPC = ArmMPController(robots[0])
            goal = np.array([2, 0, 0, 0, 0, 0, 0])
            action = np.zeros(env.n())
            k = 0
            vertices = np.array(house.Obstacles.getVertices())
            C_free = FreeSpace(vertices, [-2, 0, 0.4])
            while(1):
                ob, _, _, _ = env.step(action)
                state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
                
                if METHOD == 'Normals':
                    b, A = house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
                    # print("A: \n{}\nb: \n{}\nsides: \n{}\n".format(A, b, house.Obstacles.sides))
                    zero_col = np.zeros((b.size, 1))
                    A = np.hstack((A, zero_col))
                    
                else:
                    if (k%1 == 0):
                        p0 = [state0[0], state0[1], 0.4]
                        A, b = C_free.update_free_space(p0)
                        # C_free.show_elli(vertices, p0)
                k += 1
                #start_time = time.time()
                # try:
                #     actionMPC = MPC.solve_MPC(state0, goal, A, b)
                # except:
                #     MPC.opti.debug.show_infeasibilities()
                #     print("x: \n{}\nu :\n{}\n".format(MPC.opti.debug.value(MPC.x), MPC.opti.debug.value(MPC.u)))
                #     print("A: \n{}\nb: \n{}\n".format(A@state0[:3], b))
                #     C_free.show_elli(vertices, p0)
                # end_time = time.time()
                # print("MPC computation time: ", end_time - start_time)
                
                actionMPC = MPC.solve_MPC(state0, goal, A, b)

                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]

                # if (k%50 == 0):
                #     house.Obstacles.display()
                # k += 1
        C_free.show_elli(vertices, p0)
        env.close()