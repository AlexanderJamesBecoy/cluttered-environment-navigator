# Obstacles are being stored from house.py. If we add dynamic obstacles they need
# constatins too (house walls and doors), so it would be easier if we also do that
# in house.py

import numpy as np

class ObstacleConstraintsGenerator:
    def __init__(self, robot_dim: list, scale: float) -> None:
        self.walls = []
        self.doors = []
        self.knobs = []
        self.robot_dim = robot_dim*scale # to be used to construct constraints later
    
    def generateConstraintsCylinder(self, robot_pos: list[float], vision_range: float = 5.0) -> np.ndarray:
        """
            Generate constraints for obstacles with cylindrical collision body.
            robot_dim = [height, radius]

            returns:
                4 ndarrays with points on the left, right, lower, and upper sides of the obstacles, respectively
                The robot should then have these constraints:
                    robot_radius < left_constraints
                    robot_radius < lower_constraints
                    robot_radius > right_constraints
                    robot_radius > upper_constraints
        """
        left_constraints = []
        right_constraints = []
        lower_constraints = []
        upper_constraints = []
        center = (0, 0)
        for wall in self.walls:
            # Set center of the obstacle
            center = np.array([wall['x'], wall['y']])
            dist = np.linalg.norm(center - np.array([robot_pos[0], robot_pos[1]]))
            # Check if obstacle is out of range
            if (dist > vision_range):
                continue
            else:
                # walls were not rotated
                if np.abs(wall['theta']) != np.pi/2:
                    # Make sure constraints are feasible
                    if robot_pos[0] < center[0]:
                        left_constraints.append(wall['x'] - wall['width']/2)
                    elif robot_pos[0] > center[0]:
                        right_constraints.append(wall['x'] + wall['width']/2)
                    
                    if robot_pos[1] < center[1]:
                        lower_constraints.append(wall['y'] - wall['length']/2)
                    elif robot_pos[1] > center[1]:
                        upper_constraints.append(wall['y'] + wall['length']/2)
                else:
                    if robot_pos[0] < center[0]:
                        left_constraints.append(wall['x'] - wall['length']/2)
                    elif robot_pos[0] > center[0]:
                        right_constraints.append(wall['x'] + wall['length']/2)
                    
                    if robot_pos[1] < center[1]:
                        lower_constraints.append(wall['y'] - wall['width']/2)
                    elif robot_pos[1] > center[1]:
                        upper_constraints.append(wall['y'] + wall['width']/2)

        for door in self.doors:
            # Set center of the obstacle
            center = np.array([door['x'], door['y']])
            dist = np.linalg.norm(center - np.array([robot_pos[0], robot_pos[1]]))
            # Check if obstacle is out of range
            if (dist > vision_range):
                continue
            else:
                if np.abs(door['theta']) != np.pi/2:
                    if robot_pos[0] < center[0]:
                        left_constraints.append(door['x'] - door['width']/2)
                    elif robot_pos[0] > center[0]:
                        right_constraints.append(door['x'] + door['width']/2)
                    
                    if robot_pos[1] < center[1]:
                        lower_constraints.append(door['y'] - door['length']/2)
                    elif robot_pos[1] > center[1]:
                        upper_constraints.append(door['y'] + door['length']/2)
                else:
                    if robot_pos[0] < center[0]:
                        left_constraints.append(door['x'] - door['length']/2)
                    elif robot_pos[0] > center[0]:
                        right_constraints.append(door['x'] + door['length']/2)
                    
                    if robot_pos[1] < center[1]:
                        lower_constraints.append(door['y'] - door['width']/2)
                    elif robot_pos[1] > center[1]:
                        upper_constraints.append(door['y'] + door['width']/2)


        return np.array(left_constraints), np.array(right_constraints), np.array(lower_constraints), np.array(upper_constraints)
                