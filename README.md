# Cluttered environment navigator

This package simulates a mobile manipulator that navigates a supposedly cluttered environment. It has a sampling-based motion planner that is based on a variant of Rapidly-Exploring Random Tree (RRT*) to generate a path from one point to another in the same environment, and a Model Predictive Control (MPC) as a controller to ensure that the robot avoids obstacles.

## Contributors
- A.J. Becoy
- R. Iamoni
- Z. Lu
- G. Mussita

## Structure

This package consists of four main parts:
- Model
- Motion Planner
- Controller
- Simulation

### Model
This part describes the 7 DOF mobile manipulator and its kinematic model in the Gym environment. Files:
- **model.py**
- **kinematic.py**

### Motion Planner
This part is the sampling-based planner which generates a navigation path. File:
- **planner.py**

### Controller
This part contains the MPC which controls the mobile manipulator in order to follow the trajectory generated by the motion planner and to avoid obstacles. Files:
- **nav_MPC.py** - surface-normal MPC, avoids obstacle during navigation provided by RRT.
- **arm_MPC.py** - collision-free 3D ellipsoid MPC, computes collision-free polyhedron.
- ObstacleConstraintGenerator.py - generates the vertices, normals, etc. of obstacles obtained from class House.

### Simulation
This part contains the house environment and main files in order to run the simulation. Files:
- **nav_run.py** - simulate the mobile manipulator in the house environment with RRT* and surface-nromal MPC.
- **arm_run.py** - simulate the mobile manipulator in the test environment with collision-free 3D ellipsoid MPC.
- house.py - generates the house environment with walls, doors, and furnitures, and a test environment in a simple box with suspended osbtacle.

## Installation
```
git clone https://github.com/AlexanderJamesBecoy/door-opener-motion-planner
cd door-opener-motion-planner
```
Create an Python venv environment and install the required packages, as well as Max Spahn's gym_envs_urdf to run and simulate this project. It is require to have Python > 3.6, or < 3.10.
```
python3 -m venv env
source env/bin/activate
git clone https://github.com/maxspahn/gym_envs_urdf
cd gym_envs_urdf & pip install -e .
```
Now install the rest of the Python packages.
```
pip install casadi qpsolvers cvxpy mosek[optional]
```

If you happen to obtain an error when running the script, namely "Missing MotionPlanningGoal or MotionPlanningEnv". Run the following line in `door_motion_planner/gym_envs_urdf/`:
```
pip install -e ".[scenes]"
```

## Running the simulation
You can run the following simulations: **navigation** and **arm functionality**.
### Navigation simulation
To simulate the path-finding implementation, run the following line:
```
python3 nav_run.py
```
This displays a real-time motion of a mobile manipulator in a Gym environment, and a 2D plot of the RRT* implementation.

You can change the starting and final position by editing line 13 and 14 of **nav_run.py**:
```
INIT_POSITION = [x_i, y_i]
END_POSITION = [x_f, y_f]
```
Replace x_i, y_i, x_f and y_f for the given coordinates respectively.
**WARNING**: The values has to be within the house.

### Arm functionality
To simulate the arm obstacle avoidance, run the following line:
```
python3 arm_run.py
```

## Credits
- The furniture URDFs belong to the following websites:
    - https://github.com/personalrobotics/pr_assets
    - http://wiki.ros.org/cob_gazebo_objects