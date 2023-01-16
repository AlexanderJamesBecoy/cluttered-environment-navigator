from casadi import *
import numpy as np
from model import Model

# Default value for the cost function multipliers: these values are the same of the Max Spahn, 2021 paper
weight_tracking_default_base = 5.0
weight_tracking_default_theta = 2.0
weight_tracking_default_arm = 0.0 # 0.7
weight_input_default_base = 0.0 # 0.05
weight_input_default_theta = 0.0 # 0.05
weight_input_default_arm = 5.02
weight_terminal_default_base = 5.0
weight_terminal_default_theta = 2.0
weight_terminal_default_arm = 0.7

# Geometry parameters
offset_z = 0.4

# Denavit-Hartenberg parameters
d1 = 0.333
d3 = 0.316
d5 = 0.384
d7 = 0.107
a3 = 0.0825
a4 = 0.0825
a6 = 0.088

# Sphere constraint clearance
CLEARANCE1 = 0.5
CLEARANCE2 = 0.3
<<<<<<<<< Temporary merge branch 1
CLEARANCE3 = 0.2
=========

>>>>>>>>> Temporary merge branch 2
# MPC parameters
DT = 0.5
STEPS = 5
M = 1e6


class MPController:
    """
    This class implement a MPC for a mobile manipulator. The goal is to follow the desired path as precisely as
    possible without violating the dynamics contraints, the limits on the joints and avoiding the obstacles.
    """
    def __init__(self, model: Model, surface_dim: np.ndarray,
    weight_tracking_base: float = weight_tracking_default_base, 
    weight_tracking_theta: float = weight_tracking_default_theta, 
    weight_tracking_arm: float = weight_tracking_default_arm,
    weight_input_base: float = weight_input_default_base,
    weight_input_theta: float = weight_input_default_theta, 
    weight_input_arm: float = weight_input_default_arm,
    weight_terminal_base: float = weight_terminal_default_base,
    weight_terminal_theta: float = weight_terminal_default_theta,
    weight_terminal_arm: float = weight_terminal_default_arm,
    dt: float = DT, N: int = STEPS):
        """
        Constructor of the class.

        Args:
            model (Model): gym model of the mobile manipulator
            surface_dim: number of surfaces in the environment
            weight_tracking_base (float, optional): _description_. Defaults to weight_tracking_default_base.
            weight_tracking_theta (float, optional): _description_. Defaults to weight_tracking_default_theta.
            weight_tracking_arm (float, optional): _description_. Defaults to weight_tracking_default_arm.
            weight_input_base (float, optional): _description_. Defaults to weight_input_default_base.
            weight_input_theta (float, optional): _description_. Defaults to weight_input_default_theta.
            weight_input_arm (float, optional): _description_. Defaults to weight_input_default_arm.
            weight_terminal_base (float, optional): _description_. Defaults to weight_terminal_default_base.
            weight_terminal_theta (float, optional): _description_. Defaults to weight_terminal_default_theta.
            weight_terminal_arm (float, optional): _description_. Defaults to weight_terminal_default_arm.
            dt (float, optional): _description_. Defaults to 0.01.
            N (int, optional): _description_. Defaults to 5.
        """

        self.model = model # Model of the robot
        self.dofs = self.model._dofs # Number of dof of the robot
        self.weight_tracking = np.eye(len(self.dofs)) # Weight matrix of the error in the cost function
        self.weight_input = np.eye(len(self.dofs)) # Weight matrix of the control input in the cost function
        self.weight_terminal = np.eye(len(self.dofs)) # Weight matrix of the terminal error in the cost function

        # Assignment of the correct weights
        self.weight_tracking[0:2, :] = weight_tracking_base * self.weight_tracking[0:2, :]
        self.weight_tracking[2, :] = weight_tracking_theta * self.weight_tracking[2, :]
        self.weight_tracking[3:, :] = weight_tracking_arm * self.weight_tracking[3:, :]
        self.weight_input[0:2, :] = weight_input_base * self.weight_input[0:2, :]
        self.weight_input[2, :] = weight_input_theta * self.weight_input[2, :]
        self.weight_input[3:, :] = weight_input_arm * self.weight_input[3:, :]
        self.weight_terminal[0:2, :] = weight_terminal_base * self.weight_terminal[0:2, :]
        self.weight_terminal[2, :] = weight_terminal_theta * self.weight_terminal[2, :]
        self.weight_terminal[3:, :] = weight_terminal_arm * self.weight_terminal[3:, :]

        self.dt = dt # Time step of each iteration
        self.N = N # Prediction horizon
        self.final_cost = 0 # Total cost of the simulation

        # Limits on states and input variables
        self.lower_limit_state = self.model.get_observation_space()['joint_state']['position'].low[self.dofs]
        self.upper_limit_state = self.model.get_observation_space()['joint_state']['position'].high[self.dofs]
        self.lower_limit_input = self.model.get_observation_space()['joint_state']['velocity'].low[self.dofs]
        self.upper_limit_input = self.model.get_observation_space()['joint_state']['velocity'].high[self.dofs]
        self.surface_dim = surface_dim
        self.FHOCP()

    def FHOCP(self):
        """
            Methods to build the Finite Horizon Optimal Control Problem. Given the MPC structure, it declares the
            optimization varibles (future states and inputs), defines the cost function, and adds the control constraints.
        """

        self.opti = Opti()
        self.state0 = self.opti.parameter(len(self.dofs), 1) # Parameters for initial state
        self.goal = self.opti.parameter(len(self.dofs), 1) # Parameters for the goal state
        self.x = self.opti.variable(len(self.dofs), self.N + 1) # Optimization varibles (states) over an horizon N
        self.u = self.opti.variable(len(self.dofs), self.N) # Optimization variables (inputs) over an horizon N
        self.A = self.opti.parameter(self.surface_dim[0], self.surface_dim[1])
        self.b = self.opti.parameter(self.surface_dim[0])
        self.act = self.opti.variable(self.surface_dim[0], self.N+1)
        self.cost = 0. # Initialization of the cost function
        self.add_objective_function()
        self.opti.minimize(self.cost)
        self.add_constraints()
        p_opts = dict(print_time=False, verbose=False)
<<<<<<<<< Temporary merge branch 1
        # s_opts = dict(print_level=0, tol=5e-1, acceptable_constr_viol_tol=0.01)
        s_opts = {"max_cpu_time": 5., 
=========
        s_opts = {"max_cpu_time": 5, 
>>>>>>>>> Temporary merge branch 2
				  "print_level": 0, 
				  "tol": 5e-1, 
				  "dual_inf_tol": 5.0, 
				  "constr_viol_tol": 1e-1,
				  "compl_inf_tol": 1e-1, 
				  "acceptable_tol": 1e-2, 
				  "acceptable_constr_viol_tol": 0.01, 
				  "acceptable_dual_inf_tol": 1e10,
				  "acceptable_compl_inf_tol": 0.01,
				  "acceptable_obj_change_tol": 1e20,
				  "diverging_iterates_tol": 1e20,
                  "nlp_scaling_method": "none"}
        self.opti.solver('ipopt', p_opts, s_opts) # Set solver 'ipopt'

    def solve_MPC(self, goal: np.ndarray) -> np.ndarray:
        """
            Updates the goal and solves the optimization problem, returns the next action.
        """
        self.opti.set_value(self.goal, goal) # Set the goal state parameters
        solution = self.opti.solve()
        self.final_cost += self.opti.value(self.cost)
        return solution.value(self.u[:, 0])

    def add_objective_function(self):
        """
            Methods to build the objective function, made of three terms
            - cost to the goal (weight_tracking)
            - cost of the input (weight_input)
            - cost of the terminal point, distance from the goal at x(n+1) (weight_terminal)
        """

        for k in range(1, self.N): # Iterate over all the steps of the prediction horizon
            self.cost += (self.x[:, k] - self.goal).T @ self.weight_tracking @ (self.x[:, k] - self.goal)
            self.cost += self.u[:, k].T @ self.weight_tracking @ self.u[:, k]
        self.cost += (self.x[:, self.N] - self.goal).T @ self.weight_tracking @ (self.x[:, self.N] - self.goal)

    def add_constraints(self):
        """
            Methods to add the constraints:
            - limits on joints position (x: state)
            - limits on the joints velocity (u: input)
            - robot model kinematics/dynamics
        """

        # Limit constraints
        self.opti.subject_to(self.x[:, 0] == self.state0) # Initial state constraint
        for k in range(1, self.N): # Iterate over all the steps of the prediction horizon
            self.opti.subject_to(self.lower_limit_state <= self.x[:, k])
            self.opti.subject_to(self.x[:, k] <= self.upper_limit_state)
            self.opti.subject_to(self.lower_limit_input <= self.u[:, k])
            self.opti.subject_to(self.u[:, k] <= self.upper_limit_input)

        self.opti.subject_to(self.lower_limit_state <= self.x[:, self.N])
        self.opti.subject_to(self.x[:, self.N] <= self.upper_limit_state)

        # Robot model constraints
        for k in range(self.N):
            self.opti.subject_to(self.x[:, k+1] == self.x[:, k] + self.dt * self.u[:, k])


    def add_obstacle_avoidance_constraints(self, A, b):
        """
            Adds the obstacle avoidance constraints formulated as
                n dot p <= n dot q + M * b
                subjected to
                    sum(b) <= 3

                    n: normal vector of the surfaces of obstacles
                    p: position of the robot
                    q: point on the surface of the obstacles
                    M: a very large number
                    b: variable that controls which constraint is active. If ~1, constraint is inactive because the right hand side
                       would be very large, and thus the constraint essentially does not exist. Other other hand, if it is ~0, then
                       the constraint is active because the M is removed.

                Only at most 3 constraints should be active at once, so as to not block the robot in a box.
        """
        for k in range(self.N + 1):
            p1 = self.x[:2, k]
            self.opti.subject_to(self.A@p1 <= (self.b - CLEARANCE1 + M * self.act[:, k]))
            self.opti.subject_to(self.opti.bounded(0, self.act[:, k], 1))
            self.opti.subject_to(sum1(1-self.act[:, k]) <= 3)

<<<<<<<<< Temporary merge branch 1
            # First sphere
            p1 = [self.x[0, k], self.x[1, k], d1 + offset_z]
            # self.opti.subject_to(A @ p1 <= b - CLEARANCE1)


            for a_i, b_i in zip(A, b):
                self.opti.subject_to(a_i[0]*p1[0] + a_i[1]*p1[1] + a_i[2]*p1[2] <= b_i - CLEARANCE1)

            # Second sphere
            p2 = [self.x[0, k] - d3 * sin(self.x[3, k]) * cos(self.x[2, k]) + a3 * cos(self.x[3, k]) * cos(self.x[2, k]), \
                self.x[1, k] - d3 * sin(self.x[3, k]) * sin(self.x[2, k]) + a3 * cos(self.x[3, k])*sin(self.x[2, k]), \
                d1 + offset_z + d3 * cos(self.x[3, k]) + a3 * sin(self.x[3, k])]
            # self.opti.subject_to(A @ p2 <= b - CLEARANCE2)

            for a_i, b_i in zip(A, b):
                self.opti.subject_to(a_i[0]*p2[0] + a_i[1]*p2[1] + a_i[2]*p2[2] <= b_i - CLEARANCE2)

            # Third sphere
            p3 = [self.x[0, k] + cos(self.x[2, k]) * (d5 * (sin(self.x[4, k])*(-cos(self.x[3, k])) - cos(self.x[4, k])*sin(self.x[3, k])) \
                - d7 * (cos(self.x[5, k])*sin(self.x[4, k])*(-cos(self.x[3, k]) - cos(self.x[4, k])*sin(self.x[3, k])) \
                + sin(self.x[5, k])*(cos(self.x[4, k])*(-cos(self.x[3, k])) + sin(self.x[3, k])*sin(self.x[4, k]))) \
                - d3 * sin(self.x[3, k]) + a6 * sin(self.x[5, k])*(sin(self.x[4, k])*(-cos(self.x[3, k])) \
                - cos(self.x[4, k])*sin(self.x[3, k])) \
                - a6 * cos(self.x[5, k])*(cos(self.x[4, k])*(-cos(self.x[3, k]) + sin(self.x[3, k])*sin(self.x[4, k]))) \
                + a4 * cos(self.x[4, k])*(-cos(self.x[3, k])) \
                + a3 * cos(self.x[3, k]) \
                + a4 * sin(self.x[3, k])*sin(self.x[4, k])),
                \
                self.x[1, k] + sin(self.x[2, k]) * (d5 * (sin(self.x[4, k]) * (-cos(self.x[3, k])) \
                - cos(self.x[4, k])*sin(self.x[3, k])) \
                - d7 * (cos(self.x[5, k]) * (sin(self.x[4, k]) * (-cos(self.x[3, k])) \
                - cos(self.x[4, k]) * sin(self.x[3, k])) + sin(self.x[5, k]) * ((cos(self.x[4, k]) * (-cos(self.x[3, k])) \
                + sin(self.x[3, k]) * sin(self.x[4, k])))) \
                - d3 * sin(self.x[3, k]) \
                + a6 * sin(self.x[5, k]) * (sin(self.x[4, k]) * (-cos(self.x[3, k])) \
                - cos(self.x[4, k]) * sin(self.x[3, k])) \
                - a6 * cos(self.x[5, k]) * ((cos(self.x[4, k]) * (-cos(self.x[3, k])) \
                + sin(self.x[3, k]) * sin(self.x[4, k]))) \
                + a4 * cos(self.x[4, k]) * (-cos(self.x[3, k])) \
                + a3*cos(self.x[3, k]) + a4 * sin(self.x[3, k]) * sin(self.x[4, k])),
                \
                d1+offset_z+d7*(sin(self.x[5, k])*(1*(cos(self.x[3, k])*sin(self.x[4, k])+1*cos(self.x[4, k])*\
                sin(self.x[3, k])) - sin(self.x[3, k]) * 0 * 0) - cos(self.x[5, k]) * (cos(self.x[3, k]) * cos(self.x[4, k]) - 1 *\
                sin(self.x[3, k]) * sin(self.x[4, k]))) + d5 * (cos(self.x[3, k]) * cos(self.x[4, k]) - 1 * sin(self.x[3, k]) * sin(self.x[4, k])) + d3 *\
                cos(self.x[3, k])+a6*sin(self.x[5, k])*(cos(self.x[3, k])*cos(self.x[4, k])-1*sin(self.x[3, k])*sin(self.x[4, k]))+a3*1*\
                sin(self.x[3, k])-a4*cos(self.x[3, k])*sin(self.x[4, k])+a6*cos(self.x[5, k])*(1*(cos(self.x[3, k])*sin(self.x[4, k])+1*\
                cos(self.x[4, k]) * sin(self.x[3, k])) - sin(self.x[3, k]) * 0 * 0) - a4 * 1 * cos(self.x[4, k]) * sin(self.x[3, k])\
                ]
            
            for a_i, b_i in zip(A, b):
                self.opti.subject_to(a_i[0]*p3[0] + a_i[1]*p3[1] + a_i[2]*p3[2] <= b_i - CLEARANCE3)
=========
        self.opti.set_value(self.A, A)
        self.opti.set_value(self.b, b)
>>>>>>>>> Temporary merge branch 2
