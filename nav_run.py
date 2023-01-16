import gym
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from model import Model
from house import House
from planner import Planner
import warnings
from MPC import MPController
import time

TEST_MODE = False # Boolean to initialize test mode to test the MPC in test mode, take care that valid start and end positions are set
INIT_POSITION = [3, -3, 0.4]
END_POSITION = [-5, -3, 0.4]
NUM_RUNS = 5 # NUmber of simulations to run
STEP_SIZE = 10 # How often to compute new actions
R_SCALE = 1.0 #how much to scale the robot's dimensions for collision check
TOL = 2e-1 # Tolerance of reaching the waypoints

#Dimension of robot base, found in mobilePandaWithGripper.urdf
R_RADIUS = 0.2
R_HEIGHT = 0.3
global_time = []
global_steps = []
global_cost = []
global_T = []

def main():
    T = 0
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

        # History
        history = []

        # Set initial and end points
        start = np.array([INIT_POSITION[0], INIT_POSITION[1], 0, 0, 0, 0, 0])
        goal = np.array([END_POSITION[0], END_POSITION[1], 0, 0, 0, 0, 0])

        # Generate the environment
        start_pos = robots[0].set_initial_pos(INIT_POSITION[:2])
        ob = env.reset(pos=start_pos)
        is_open = {
            'bathroom':         True,
            'outdoor':          True,
            'top_bedroom':      True,
            'bottom_bedroom':   True,
            'kitchen':          True,
        }

        house.generate_walls()
        house.generate_furniture()
        if not TEST_MODE:
            house.generate_doors()

        planner = Planner(house=house, test_mode=TEST_MODE, debug_mode=False)
        no_rooms = planner.plan_motion(INIT_POSITION[:2], END_POSITION[:2], step_size=1)
        house.draw_walls()
        house.draw_furniture()
        if not TEST_MODE:
            house.draw_doors(is_open)

        # Initialize MPC controller
        b, A = house.Obstacles.generateConstraintsCylinder() # Compute the normals and offsets of the walls
        MPC = MPController(robots[0], A.shape)
        action = np.zeros(env.n())

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

        # Initial step and observation
        ob, _, _, _ = env.step(action)
        state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
        p0 = [state0[0], state0[1], 0.4]

        # Set initial MPC variables and constraint parameters
        MPC.opti.set_initial(MPC.x[:, 0], state0)
        MPC.add_obstacle_avoidance_constraints(A, b)
        MPC.opti.set_value(MPC.state0, state0)

        t = 0
        start_time = time.time()
        for waypoint in route:
            goal = np.array([waypoint[0], waypoint[1], 0, 0, 0, 0, 0])

            while (1):
                ob, _, _, _ = env.step(action)
                state0 = ob['robot_0']['joint_state']['position'][robots[0]._dofs]
                MPC.opti.set_value(MPC.state0, state0)
                p0 = [state0[0], state0[1], 0.4]
                
                if (np.allclose(p0, waypoint, rtol=TOL, atol=TOL)):
                    print("Point reached")
                    MPC.opti.set_value(MPC.state0, state0)
                    break

                if (t%STEP_SIZE == 0):
                    # Compute the next action
                    actionMPC = MPC.solve_MPC(goal)

                action = np.zeros(env.n())
                for i, j in enumerate(robots[0]._dofs):
                    action[j] = actionMPC[i]

                t += 1
        end_time = time.time()
        print("Elapsed time: {}\nCost: {}\nSteps: {}".format(end_time - start_time, MPC.final_cost, t))
        global_time.append(end_time - start_time)
        global_steps.append(t)
        global_cost.append(MPC.final_cost)
        print("GOAL REACHED!")
        global_T.append(T)
        env.close()

if __name__ == "__main__":
    for i in range(NUM_RUNS):
        main()

    print("Performance statistics: \n")
    print("Times per simulation: {}\nCost of simulation: {}\nTotal steps per simulation: {}\n".format(global_time, global_cost, global_steps))
    print("Average time per step: {}\nAverage cost: {}\n".format(np.sum(np.array(global_time)/np.array(global_steps))/NUM_RUNS, np.sum(global_cost)/NUM_RUNS))
    