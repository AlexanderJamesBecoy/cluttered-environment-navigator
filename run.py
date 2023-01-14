import gym
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from model import Model
from house import House
from planner import Planner
import warnings
from MPC import MPController
from free_space import FreeSpace
import matplotlib.pyplot as plt
from shapely import Polygon
from scipy.spatial import ConvexHull

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
        house = House(env, robot_dim=robot_dim, scale=R_SCALE, test_mode=True)
        ellipsoids = []

        # History
        history = []

        # Set initial and end points
        init_position = [-3, -3, 0.4]
        end_position = [3, 3, 0.4]

        # Generate the environment
        start_pos = robots[0].set_initial_pos(init_position[:2])
        ob = env.reset(pos=start_pos)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }
        house.generate_walls()
        # house.generate_doors()
        house.generate_furniture()
        planner = Planner(house=house, test_mode=True, debug_mode=False)
        no_rooms = planner.plan_motion(init_position[:2], end_position[:2], step_size = 1)
        house.draw_walls()
        # house.draw_doors(is_open)
        house.draw_furniture()

        # Initialize MPC controller and FreeSpace for ellipsoid calculations
        start = np.array([init_position[0], init_position[1], 0, 0, 0, 0, 0])
        goal = np.array([end_position[0], end_position[1], 0, 0, 0, 0, 0])
        MPC = MPController(robots[0])
        action = np.zeros(env.n())
        k = 0
        vertices = np.array(house.Obstacles.getVertices())
        C_free = FreeSpace(vertices, init_position)

        # Combine the routes
        route = []
        for i, points in enumerate(planner._routes):
            if i == 0:
                route = np.array(points)
            elif i < len(planner._routes):
                route = np.concatenate((route, points), axis=0)

        # Add a column of z-offsets
        z_col = np.ones((route.shape[0], 1)) * 0.4
        route = np.hstack((route, z_col))
        planner.plot_plan_2d(0)
        # Calculate all ellipsoids
        print("---------------------------")
        print("Computing ellipsoids...")

        for i, waypoint in enumerate(route):
            A, b = C_free.update_free_space(waypoint)
            A = np.array(A)
            ellipsoids.append([A, b, C_free.ellipsoid.C, C_free.ellipsoid.d])
            
            # C_free.show_elli(vertices, waypoint, end_position)
        print("Finished computing ellipsoids")
        indices = [i for i in range(len(ellipsoids))]

        # Initial step and observation
        action = np.zeros(env.n())
        # state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
        # p0 = [state0[0], state0[1], 0.4]
        # planner.plot_plan_2d(0)
        # MPC.opti.set_initial(MPC.x[:, 0], start)
        for ellipsoid, waypoint, k in zip(ellipsoids, route, indices):
            goal = np.array([waypoint[0], waypoint[1], 0, 0, 0, 0, 0])

            if (k > 0):
                A = np.concatenate((ellipsoid[0], ellipsoids[k-1][0]), axis=0)
                b = np.concatenate((ellipsoid[1], ellipsoids[k-1][1]), axis=0)

                # # Remove overlapping parts
                # b_int1 = np.rint(ellipsoid[1])
                # b_int2 = np.rint(ellipsoids[i-1][1])
                # common_elements = set(b_int1).intersection(b_int2)
                # indices_1 = [list(b_int1).index(x) for x in common_elements]
                # indices_2 = [len(ellipsoid[0])+list(b_int2).index(x) for x in common_elements]

                # for index in sorted(indices_2, reverse=True):
                #     A = np.delete(A, index, axis=0)
                #     b = np.delete(b, index)
                
                # for index in sorted(indices_1, reverse=True):
                #     A = np.delete(A, index, axis = 0)
                #     b = np.delete(b, index)
                A_i = []
                b_i = []

                for a, c in zip(A, b):
                    if (a@waypoint < c) and (a@route[k-1] < c):
                        A_i.append(a)
                        b_i.append(c)
                
                A = np.array(A_i)
                b = np.array(b_i)
            else:
                A = ellipsoid[0]
                b = ellipsoid[1]

            # MPC.add_obstacle_avoidance_constraints(A, b) 

            while (1):
                ob, _, _, _ = env.step(action)
                state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]

                p0 = [state0[0], state0[1], 0.4]
                print("current point")
                print(k, p0, waypoint)
                    
                if (np.allclose(p0, waypoint, rtol=TOL, atol=TOL)):
                    print(k, p0, waypoint)
                    print("Point reached")
                    break

                try:
                    actionMPC = MPC.solve_MPC(state0, goal, A, b)
                except:
                    print(MPC.opti.debug.show_infeasibilities())
                    print(MPC.opti.debug.value(MPC.x[:, 0]))
                    print(MPC.opti.debug.value(MPC.state0))
                    C_free.ellipsoid.C = ellipsoid[2]
                    C_free.ellipsoid.d = ellipsoid[3]
                    C_free.show_elli(vertices, p0, waypoint)

                    C_free.ellipsoid.C = ellipsoids[k-1][2]
                    C_free.ellipsoid.d = ellipsoids[k-1][3]
                    C_free.show_elli(vertices, p0, route[k-1])
                # actionMPC = MPC.solve_MPC(state0, goal, ellipsoid[0], ellipsoid[1])
                # If current target waypoint is not the last one
                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]
                # MPC.opti.set_value(MPC.state0, state0)
            
                MPC.refresh_MPC()
        
        print("GOAL REACHED!")
        env.close()