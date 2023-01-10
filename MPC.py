from casadi import *
import numpy as np
from model import Model

# Default value for the cost function multipliers: these values are the same of the Max Spahn, 2021 paper
weight_tracking_default_base = 5.0
weight_tracking_default_theta = 2.0
weight_tracking_default_arm = 0.7
weight_input_default_base = 0.05
weight_input_default_theta = 0.05
weight_input_default_arm = 5.02
weight_terminal_default_base = 5.0
weight_terminal_default_theta = 2.0
weight_terminal_default_arm = 0.7


class MPController:
    """
    This class implement a MPC for a mobile manipulator. The goal is to follow the desired path as precisely as
    possible without violating the dynamics contraints, the limits on the joints and avoiding the obstacles.
    """
    def __init__(self, model: Model, weight_tracking_base: float = weight_tracking_default_base, 
    weight_tracking_theta: float = weight_tracking_default_theta, 
    weight_tracking_arm: float = weight_tracking_default_arm,
    weight_input_base: float = weight_input_default_base,
    weight_input_theta: float = weight_input_default_theta, 
    weight_input_arm: float = weight_input_default_arm,
    weight_terminal_base: float = weight_terminal_default_base,
    weight_terminal_theta: float = weight_terminal_default_theta,
    weight_terminal_arm: float = weight_terminal_default_arm,
    dt: float = 0.01, N: int = 5):
        """
        Constructor of the classe.

        Args:
            model (Model): gym model of the mobile manipulator
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

        # Limits on states and input variables
        self.lower_limit_state = self.model.get_observation_space()['joint_state']['position'].low[self.dofs]
        self.upper_limit_state = self.model.get_observation_space()['joint_state']['position'].high[self.dofs]
        self.lower_limit_input = self.model.get_observation_space()['joint_state']['velocity'].low[self.dofs]
        self.upper_limit_input = self.model.get_observation_space()['joint_state']['velocity'].high[self.dofs]

        self.FHOCP()

    def FHOCP(self):
        """
        Methods to build the Finite Horizon Optimal Control Problem. Given the MPC structure, it declares the
        optimization varibles (future states and inputs), it defines the cost function, it adds the constraints
        and it solves the optimization problem.
        """

        self.opti = Opti()
        self.state0 = self.opti.parameter(len(self.dofs), 1) # Parameters for initial state
        self.goal = self.opti.parameter(len(self.dofs), 1) # Parameters for the goal state
        self.x = self.opti.variable(len(self.dofs), self.N + 1) # Optimization varibles (states) over an horizon N
        self.u = self.opti.variable(len(self.dofs), self.N) # Optimization variables (inputs) over an horizon N
        self.cost = 0. # Initialization of the cost function
        self.add_objective_function()
        self.opti.minimize(self.cost)
        self.add_constraints()
        p_opts = dict(print_time=True, verbose=False)
        s_opts = dict(print_level=0)
        self.opti.solver('ipopt', p_opts, s_opts) # Set solver 'ipopt'
        # solution = self.opti.solve() # Solve the problem
        # print(solution.value(self.u[:, 0]))

        # return solution.value(self.u[:, 0])

    def solve_MPC(self, state0: np.ndarray, goal: np.ndarray) -> np.ndarray:

        self.opti.set_value(self.state0, state0)
        self.opti.set_value(self.goal, goal)
        solution = self.opti.solve() # Solve the problem

        return solution.value(self.u[:, 0])

    def add_objective_function(self):
        """
        Methods to build the objective function, made of three terms
        - cost to the goal (weight_tracking)
        - cost of the input (weight_input)
        - cost of the terminal point, distance from the goal at x(n+1) (weight_terminal)
        """

        for k in range(self.N): # Iterate over all the steps of the prediction horizon
            self.cost += (self.x[:, k] - self.goal).T @ self.weight_tracking @ (self.x[:, k] - self.goal)
            self.cost += self.u[:, k].T @ self.weight_tracking @ self.u[:, k]
        self.cost += (self.x[:, self.N] - self.goal).T @ self.weight_tracking @ (self.x[:, self.N] - self.goal)

    def add_constraints(self):
        """
        Methods to add the constraints:
        - limits on joints position (x: state)
        - limits on the joints velocity (u: input)
        - robot model kinematics/dynamics
        - static ostable avoidance
        """
        self.opti.subject_to(self.x[:, 0] == self.state0) # Initial state constraint
        for k in range(self.N): # Iterate over all the steps of the prediction horizon
            self.opti.subject_to(self.lower_limit_state <= self.x[:, k])
            self.opti.subject_to(self.x[:, k] <= self.upper_limit_state)
            self.opti.subject_to(self.lower_limit_input <= self.u[:, k])
            self.opti.subject_to(self.u[:, k] <= self.upper_limit_input)

        self.opti.subject_to(self.lower_limit_state <= self.x[:, self.N])
        self.opti.subject_to(self.x[:, self.N] <= self.upper_limit_state)

        # Robot model constraints

        for k in range(self.N):
            self.opti.subject_to(self.x[:, k+1] == self.x[:, k] + self.dt * self.u[:, k])
        # Static obstacles avoidance: TODO
