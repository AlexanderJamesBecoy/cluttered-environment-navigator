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
METHOD = ''
TOL = 1e-1

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
        house = House(env, robot_dim=robot_dim, scale=R_SCALE, test_mode=False)
        house.generate_walls()
        house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=False)
        no_rooms = planner.plan_motion()

        # History
        history = []

        # Generate environment
        # route, open = planner.generate_waypoints(room)
        # init_joints = robots[0].set_initial_pos(route[0])
        start_pos = robots[0].set_initial_pos([3, -3])
        ob = env.reset(pos=start_pos)
        house.draw_walls()
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }
        house.draw_doors(is_open)
        house.draw_furniture()
        # planner.plot_plan_2d(route)

        # Follow a path set by waypoints   z
        MPC = MPController(robots[0])
        init_point = [3, -3, 0.4]
        end_point = [-5, -3, 0.4]
        goal = np.array([end_point[0], end_point[1], 0, 0, 0, 0, 0])
        action = np.zeros(env.n())
        k = 0
        vertices = np.array(house.Obstacles.getVertices())
        # print("All vertices: \n{}\n".format(vertices))
        C_free = FreeSpace(vertices, init_point)

        while(1):
            ob, _, _, _ = env.step(action)
            state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]

            if METHOD == 'Normals':
                b, A = house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
                # print("A: \n{}\nb: \n{}\nsides: \n{}\n".format(A, b, house.Obstacles.sides))
                zero_col = np.zeros((b.size, 1))
                A = np.hstack((A, zero_col))
                
            else:
                house.Obstacles.generateConstraintsCylinder(ob['robot_0']['joint_state']['position'])
                p0 = [state0[0], state0[1], 0.4]
                if (k%100 == 0):
                    A, b = C_free.update_free_space(p0)
                    C_free.show_elli(vertices, p0, end_point)
                if np.allclose(p0, end_point, rtol=TOL, atol=TOL):
                    C_free.show_elli(vertices, p0, end_point)
                    print("\nFINISHED!\n")
                    break

                #start_time = time.time()
                try:
                    actionMPC = MPC.solve_MPC(state0, goal, A, b)
                except:
                    MPC.opti.debug.show_infeasibilities()
                    C_free.show_elli(vertices, p0, end_point)
                #end_time = time.time()
                #print("MPC computation time: ", end_time - start_time)

                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]
            k += 1
            # if (k%50 == 0):
            #     house.Obstacles.display()
            # k += 1 
    env.close()
        