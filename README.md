# Cluttered environment navigator

TODO: description

## Installation
```
git clone https://github.com/AlexanderJamesBecoy/door-opener-motion-planner
cd door-opener-motion-planner
```
Create an Python venv environment and install the required packages, as well as Max Spahn's gym_envs_urdf to run and simulate this project. It is require to have Python > 3.6, or < 3.10.
```
python3 -m venv env
git clone https://github.com/maxspahn/gym_envs_urdf
cd gym_envs_urdf & pip install -e .
```

```
TODO: missing packages
```

If you happen to obtain an error when running the script, namely "Missing MotionPlanningGoal or MotionPlanningEnv". Run the following line in `door_motion_planner/gym_envs_urdf`:
```
pip install -e ".[scenes]"
```

## Running the simulation


## Credits
- The furniture URDFs belong to the following websites:
    - https://github.com/personalrobotics/pr_assets
    - http://wiki.ros.org/cob_gazebo_objects