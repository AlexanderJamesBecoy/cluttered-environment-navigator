import numpy as np
import cvxpy as cp
from model import Model

class MPController:
    def __init__(self, model: Model, weight_tracking: np.ndarray, weight_input: np.ndarray, weight_terminal: np.ndarray,
    upper_limit_state: float, lower_limit_state: float, upper_limit_input: float, lower_limit_input: float, 
    dt: float = 0.001, N: int = 5):

        self.model = model # Model of the robot
        self.dof = len(self.model.dofs) # Number of dof of the robot
        self.weight_tracking = weight_tracking # Weight matrix of the error in the cost function
        self.weight_input = weight_input # Weight matrix of the control input in the cost function
        self.weight_terminal = weight_terminal # # Weight matrix of the terminal error in the cost function
        self.dt = dt # Time step of each iteration
        self.N = N # Prediction horizon
        self.upper_limit_state = upper_limit_state
        self.lower_limit_state = lower_limit_state
        self.upper_limit_input = upper_limit_input
        self.lower_limit_input = lower_limit_input

    def FHOCP(self, state0: np.ndarray, goal: np.ndarray): # Finite Horizon Optimal Control Problem

        self.x = cp.variables((self.dof, self.N + 1)) # Optimization varibles (states) over an horizon N
        self.u = cp.variables((self.dof, self.N)) # Optimization variables (inputs) over an horizon N
        self.cost = 0. # Initialization of the cost function
        self.constraints = [] # Initialization of the constraints
        self.add_objective_function(state0, goal)
        self.add_constraints(state0, goal)

        problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        problem.solve(solver=cp.OSQP)

        return 

    def add_objective_function(self, state0: np.ndarray, goal: np.ndarray):

        for k in range(self.N): # Iterate over all the steps of the prediction horizon
            self.cost += cp.quad_form(self.x[:, k] - goal, self.weight_tracking)
            self.cost += cp.quad_form(self.u[:, k], self.weight_input)
        self.cost += cp.quad_form(self.x[:, self.N] - goal, self.weight_terminal)

    def add_constraints(self, state0):

        self.constraints += [self.z[:, 0] == state0] # Initial state constraint
        for k in range(self.N): # Iterate over all the steps of the prediction horizon
            self.constraints += [self.lower_limit_state <= self.x[:, k] <= self.upper_limit_state]
            self.constraints += [self.lower_limit_input <= self.u[:, k] <= self.upper_limit_input]
        self.constraints += [self.lower_limit_state <= self.x[:, self.N] <= self.upper_limit_state]
