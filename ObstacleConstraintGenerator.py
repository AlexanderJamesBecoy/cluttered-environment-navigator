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

    def computeNormalVector(self, p1: list[float, float], p2: list[float, float]) -> list[float, float]:
        """
            Returns the normal vector of the line defined by points p1 and p2.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return [dy, -dx], [-dy, dx] # [dy, -dx] -> left and top side of obstacle, [-dy, dx] -> right and lower side of the obstacle
    
    def generateConstraintsCylinder(self, robot_pos: list[float], vision_range: float = 5.0) -> np.ndarray:
        """
            Generate constraints for obstacles with cylindrical collision body.
            robot_dim = [height, radius]

            returns:
                5 ndarrays of dot products of robot position and normals of sides of the obstacles, and dot products of obstacle side positions and normals of sides of the obstacles.
                The robot should then have these constraints:
                    robot_norms[i] < constraints - radius
        """
        left_constraints = []
        right_constraints = []
        lower_constraints = []
        upper_constraints = []
        robot_norms = []
        center = (0, 0)
        for wall in self.walls:
            # Set center of the obstacle
            center = np.array([wall['x'], wall['y']])
            dist = np.linalg.norm(center - np.array([robot_pos[0], robot_pos[1]]))
            # Check if obstacle is out of range, can be improved by checking each side but takes more time
            if (dist > vision_range):
                continue
            else:
                # walls were not rotated
                if np.abs(wall['theta']) != np.pi/2:
                    # Compute the corner locations and center of each side
                    left_point = [center[0] - wall['width']/2, center[1]]
                    top_point = [center[0], center[1] + wall['length']/2]
                    right_point = [center[0] + wall['width']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['length']]

                    tl = [wall['x'] - wall['width']/2, wall['y'] + wall['length']/2]
                    tr = [wall['x'] + wall['width']/2, wall['y'] + wall['length']/2]
                    br = [wall['x'] + wall['width']/2, wall['y'] - wall['length']/2]
                    bl = [wall['x'] - wall['width']/2, wall['y'] - wall['length']/2]
                else:
                    left_point = [center[0] - wall['length']/2, center[1]]
                    top_point = [center[0], center[1] + wall['width']/2]
                    right_point = [center[0] + wall['length']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['width']]

                    tl = [wall['x'] - wall['length']/2, wall['y'] + wall['width']/2]
                    tr = [wall['x'] + wall['length']/2, wall['y'] + wall['width']/2]
                    br = [wall['x'] + wall['length']/2, wall['y'] - wall['width']/2]
                    bl = [wall['x'] - wall['length']/2, wall['y'] - wall['width']/2]
                
                # Compute the normal vectors on each side
                left_norm = self.computeNormalVector(bl, tl)[0]
                top_norm = self.computeNormalVector(tl, tr)[0]
                right_norm = self.computeNormalVector(br, tr)[1]
                bot_norm = self.computeNormalVector(bl, br)[1]

                # Transform to unit vectors
                left_norm = left_norm / np.linalg.norm(left_norm)
                top_norm = top_norm / np.linalg.norm(top_norm)
                right_norm = right_norm / np.linalg.norm(right_norm)
                bot_norm = bot_norm / np.linalg.norm(bot_norm)

                # Check which constraints should be active, append those to the final lists
                # Constrain is active if the robot is on that side of the obstacle. If it's diagonal to the obstacle, then multiple constraints are active
                if robot_pos[0] < left_point[0]: # left side of obstacle
                    robot_norms.append(left_norm@robot_pos[:2])
                    left_constraints.append(left_norm@left_point)

                    if robot_pos[1] < bot_point[1]:
                        robot_norms.append(bot_norm@robot_pos[:2])
                        lower_constraints.append(bot_norm@bot_point)
                    elif robot_pos[1] > top_point[1]:
                        robot_norms.append(top_norm@robot_pos[:2])
                        upper_constraints.append(top_norm@top_point)
                elif robot_pos[0] > right_point[0]: # right side of obstacle
                    robot_norms.append(right_norm@robot_pos[:2])
                    right_constraints.append(right_norm@right_point)

                    if robot_pos[1] < bot_point[1]:
                        robot_norms.append(bot_norm@robot_pos[:2])
                        lower_constraints.append(bot_norm@bot_point)
                    elif robot_pos[1] > top_point[1]:
                        robot_norms.append(top_norm@robot_pos[:2])
                        upper_constraints.append(top_norm@top_point)
                elif robot_pos[1] < bot_point[1]: # bottom side of obstacle
                    robot_norms.append(bot_norm@robot_pos[:2])
                    lower_constraints.append(bot_norm@bot_point)

                    if robot_pos[0] < left_point[0]:
                        robot_norms.append(left_norm@robot_pos[:2])
                        left_constraints.append(left_norm@left_point)
                    elif robot_pos[0] > right_point[0]:
                        robot_norms.append(right_norm@robot_pos[:2])
                        right_constraints.append(right_norm@right_point)
                
                elif robot_pos[1] > top_point[1]:
                    robot_norms.append(top_norm@robot_pos[:2])
                    upper_constraints.append(top_norm@top_point)

                    if robot_pos[0] < left_point[0]:
                        robot_norms.append(left_norm@robot_pos[:2])
                        left_constraints.append(left_norm@left_point)
                    elif robot_pos[0] > right_point[0]:
                        robot_norms.append(right_norm@robot_pos[:2])
                        right_constraints.append(right_norm@right_point)

        for door in self.doors:
            # Set center of the obstacle
            center = np.array([door['x'], door['y']])
            dist = np.linalg.norm(center - np.array([robot_pos[0], robot_pos[1]]))
            # Check if obstacle is out of range
            if (dist > vision_range):
                continue
            else:
                # walls were not rotated
                if np.abs(wall['theta']) != np.pi/2:
                    # Compute the corner locations and center of each side
                    left_point = [center[0] - wall['width']/2, center[1]]
                    top_point = [center[0], center[1] + wall['length']/2]
                    right_point = [center[0] + wall['width']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['length']]

                    tl = [wall['x'] - wall['width']/2, wall['y'] + wall['length']/2]
                    tr = [wall['x'] + wall['width']/2, wall['y'] + wall['length']/2]
                    br = [wall['x'] + wall['width']/2, wall['y'] - wall['length']/2]
                    bl = [wall['x'] - wall['width']/2, wall['y'] - wall['length']/2]
                else:
                    left_point = [center[0] - wall['length']/2, center[1]]
                    top_point = [center[0], center[1] + wall['width']/2]
                    right_point = [center[0] + wall['length']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['width']]

                    tl = [wall['x'] - wall['length']/2, wall['y'] + wall['width']/2]
                    tr = [wall['x'] + wall['length']/2, wall['y'] + wall['width']/2]
                    br = [wall['x'] + wall['length']/2, wall['y'] - wall['width']/2]
                    bl = [wall['x'] - wall['length']/2, wall['y'] - wall['width']/2]
                
                # Compute the normal vectors on each side
                left_norm = self.computeNormalVector(bl, tl)[0]
                top_norm = self.computeNormalVector(tl, tr)[0]
                right_norm = self.computeNormalVector(br, tr)[1]
                bot_norm = self.computeNormalVector(bl, br)[1]

                # Transform to unit vectors
                left_norm = left_norm / np.linalg.norm(left_norm)
                top_norm = top_norm / np.linalg.norm(top_norm)
                right_norm = right_norm / np.linalg.norm(right_norm)
                bot_norm = bot_norm / np.linalg.norm(bot_norm)

                # Check which constraints should be active, append those to the final lists
                # Constrain is active if the robot is on that side of the obstacle. If it's diagonal to the obstacle, then multiple constraints are active
                if robot_pos[0] < left_point[0]: # left side of obstacle
                    robot_norms.append(left_norm@robot_pos[:2])
                    left_constraints.append(left_norm@left_point)

                    if robot_pos[1] < bot_point[1]:
                        robot_norms.append(bot_norm@robot_pos[:2])
                        lower_constraints.append(bot_norm@bot_point)
                    elif robot_pos[1] > top_point[1]:
                        robot_norms.append(top_norm@robot_pos[:2])
                        upper_constraints.append(top_norm@top_point)
                elif robot_pos[0] > right_point[0]: # right side of obstacle
                    robot_norms.append(right_norm@robot_pos[:2])
                    right_constraints.append(right_norm@right_point)

                    if robot_pos[1] < bot_point[1]:
                        robot_norms.append(bot_norm@robot_pos[:2])
                        lower_constraints.append(bot_norm@bot_point)
                    elif robot_pos[1] > top_point[1]:
                        robot_norms.append(top_norm@robot_pos[:2])
                        upper_constraints.append(top_norm@top_point)
                elif robot_pos[1] < bot_point[1]: # bottom side of obstacle
                    robot_norms.append(bot_norm@robot_pos[:2])
                    lower_constraints.append(bot_norm@bot_point)

                    if robot_pos[0] < left_point[0]:
                        robot_norms.append(left_norm@robot_pos[:2])
                        left_constraints.append(left_norm@left_point)
                    elif robot_pos[0] > right_point[0]:
                        robot_norms.append(right_norm@robot_pos[:2])
                        right_constraints.append(right_norm@right_point)
                
                elif robot_pos[1] > top_point[1]:
                    robot_norms.append(top_norm@robot_pos[:2])
                    upper_constraints.append(top_norm@top_point)

                    if robot_pos[0] < left_point[0]:
                        robot_norms.append(left_norm@robot_pos[:2])
                        left_constraints.append(left_norm@left_point)
                    elif robot_pos[0] > right_point[0]:
                        robot_norms.append(right_norm@robot_pos[:2])
                        right_constraints.append(right_norm@right_point)


        return np.array(robot_norms), np.array(left_constraints), np.array(right_constraints), np.array(lower_constraints), np.array(upper_constraints)
                