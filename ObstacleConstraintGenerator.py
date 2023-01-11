# Obstacles are being stored from house.py. If we add dynamic obstacles they need
# constatins too (house walls and doors), so it would be easier if we also do that
# in house.py

import numpy as np
import matplotlib.pyplot as plt

class ObstacleConstraintsGenerator:
    def __init__(self, robot_dim: list, scale: float) -> None:
        self.walls = []
        self.doors = []
        self.furnitures = []
        self.knobs = []
        self.robot_dim = robot_dim*scale # to be used to construct constraints later
        self.robot_norms = []
        self.constraints = []
        self.vectors_doors = []
        self.vectors_walls = []
        self.robot_pos = 0;
        self.points_doors = []
        self.points_walls = []
        self.normals = []

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
                robot_norms: ndarray of dot products of normal vectors of the sides of the obstacles and robot positions
                constraints: ndarray of dot products of normal vectors of the sides of the obstacles and position of the sides
            
            To apply the constraints, simply do:
                robot_norms[i] < constraints[i]
            
            You can also include a radius to keep the robot some distance away from the walls like so:
                robot_norms[i] < constraints[i] - offset
            
            The code checks which side the robot is on and activate the appropriate constraint at each time step.
        """
        constraints = []
        robot_norms = []
        r = 0.2
        center = (0, 0)
        self.vectors_walls = []
        self.points_walls = []
        self.vectors_doors = []
        self.points_doors = []
        self.normals = []

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
                    bot_point = [center[0], center[1] - wall['length']/2]

                    tl = [wall['x'] - wall['width']/2, wall['y'] + wall['length']/2]
                    tr = [wall['x'] + wall['width']/2, wall['y'] + wall['length']/2]
                    br = [wall['x'] + wall['width']/2, wall['y'] - wall['length']/2]
                    bl = [wall['x'] - wall['width']/2, wall['y'] - wall['length']/2]
                else:
                    left_point = [center[0] - wall['length']/2, center[1]]
                    top_point = [center[0], center[1] + wall['width']/2]
                    right_point = [center[0] + wall['length']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['width']/2]

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
                                
                self.normals.append(left_norm)
                self.normals.append(right_norm)
                self.normals.append(top_norm)
                self.normals.append(bot_norm)

                # Check which constraints should be active, append those to the final lists
                # Constrain is active if the robot is on that side of the obstacle. If it's diagonal to the obstacle, then multiple constraints are active
                print('Walls')
                if robot_pos[0] < left_point[0]: # left side of obstacle
                    robot_norms.append(left_norm@robot_pos[:2])
                    constraints.append(left_norm@left_point)
                    self.vectors_walls.append(center-left_point)
                    self.points_walls.append(left_point)

                    if robot_pos[1] < bot_point[1]:
                        print('left bot')
                        robot_norms.append(bot_norm@robot_pos[:2])
                        constraints.append(bot_norm@bot_point)
                        self.vectors_walls.append(center-bot_point)
                        self.points_walls.append(bot_point)
                    elif robot_pos[1] > top_point[1]:
                        print('left top')
                        robot_norms.append(top_norm@robot_pos[:2])
                        constraints.append(top_norm@top_point)
                        self.vectors_walls.append(center-top_point)
                        self.points_walls.append(top_point)

                elif robot_pos[0] > right_point[0]: # right side of obstacle
                    robot_norms.append(right_norm@robot_pos[:2])
                    constraints.append(right_norm@right_point)
                    print('right')
                    self.vectors_walls.append(center-right_point)
                    self.points_walls.append(right_point)
                    
                    if robot_pos[1] < bot_point[1]:
                        print('right bot')
                        robot_norms.append(bot_norm@robot_pos[:2])
                        constraints.append(bot_norm@bot_point)
                        self.vectors_walls.append(center-bot_point)
                        self.points_walls.append(bot_point)
                    elif robot_pos[1] > top_point[1]:
                        print('right top')
                        robot_norms.append(top_norm@robot_pos[:2])
                        constraints.append(top_norm@top_point)
                        self.vectors_walls.append(center-top_point)
                        self.points_walls.append(top_point)

                elif robot_pos[1] < bot_point[1]: # bottom side of obstacle
                    robot_norms.append(bot_norm@robot_pos[:2])
                    constraints.append(bot_norm@bot_point)
                    print('bot')
                    self.vectors_walls.append(center-bot_point)
                    self.points_walls.append(bot_point)
                    
                    if robot_pos[0] < left_point[0]:
                        print('bot left')
                        robot_norms.append(left_norm@robot_pos[:2])
                        constraints.append(left_norm@left_point)
                        self.vectors_walls.append(center-left_point)
                        self.points_walls.append(left_point)
                    elif robot_pos[0] > right_point[0]:
                        print('bot right')
                        robot_norms.append(right_norm@robot_pos[:2])
                        constraints.append(right_norm@right_point)
                        self.vectors_walls.append(center-right_point)
                        self.points_walls.append(right_point)
                    
                elif robot_pos[1] > top_point[1]:
                    print('top')
                    robot_norms.append(top_norm@robot_pos[:2])
                    constraints.append(top_norm@top_point)
                    self.vectors_walls.append(center-top_point)
                    self.points_walls.append(top_point)
                    
                    if robot_pos[0] < left_point[0]:
                        print('top left')
                        robot_norms.append(left_norm@robot_pos[:2])
                        constraints.append(left_norm@left_point)
                        self.vectors_walls.append(center-left_point)
                        self.points_walls.append(left_point)
                    elif robot_pos[0] > right_point[0]:
                        print('top right')
                        robot_norms.append(right_norm@robot_pos[:2])
                        constraints.append(right_norm@right_point)
                        self.vectors_walls.append(center-right_point)
                        self.points_walls.append(right_point)
                    
        for door in self.doors:
            # Set center of the obstacle
            center = np.array([door['x'], door['y']])
            dist = np.linalg.norm(center - np.array([robot_pos[0], robot_pos[1]]))
            # Check if obstacle is out of range
            if (dist > vision_range):
                continue
            else:
                # walls were not rotated
                if np.abs(door['theta']) == np.pi/2:
                    # Compute the corner locations and center of each side
                    left_point = [center[0] - wall['width']/2, center[1]]
                    top_point = [center[0], center[1] + wall['length']/2]
                    right_point = [center[0] + wall['width']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['length']/2]

                    tl = [wall['x'] - wall['width']/2, wall['y'] + wall['length']/2]
                    tr = [wall['x'] + wall['width']/2, wall['y'] + wall['length']/2]
                    br = [wall['x'] + wall['width']/2, wall['y'] - wall['length']/2]
                    bl = [wall['x'] - wall['width']/2, wall['y'] - wall['length']/2]
                else:
                    left_point = [center[0] - wall['length']/2, center[1]]
                    top_point = [center[0], center[1] + wall['width']/2]
                    right_point = [center[0] + wall['length']/2, center[1]]
                    bot_point = [center[0], center[1] - wall['width']/2]

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
                
                self.normals.append(left_norm)
                self.normals.append(right_norm)
                self.normals.append(top_norm)
                self.normals.append(bot_norm)
                # Check which constraints should be active, append those to the final lists
                # Constrain is active if the robot is on that side of the obstacle. If it's diagonal to the obstacle, then multiple constraints are active
                print('Doors')
                if robot_pos[0] < left_point[0]: # left side of obstacle
                    robot_norms.append(left_norm@robot_pos[:2])
                    constraints.append(left_norm@left_point)
                    print('left')
                    self.vectors_doors.append(center-left_point)
                    self.points_doors.append(left_point)

                    if robot_pos[1] < bot_point[1]:
                        print('left bot')
                        robot_norms.append(bot_norm@robot_pos[:2])
                        constraints.append(bot_norm@bot_point)
                        self.vectors_doors.append(center-bot_point)
                        self.points_doors.append(bot_point)
                    
                    elif robot_pos[1] > top_point[1]:
                        print('left top')
                        robot_norms.append(top_norm@robot_pos[:2])
                        constraints.append(top_norm@top_point)
                        self.vectors_doors.append(center-top_point)
                        self.points_doors.append(top_point)
                
                elif robot_pos[0] > right_point[0]: # right side of obstacle
                    robot_norms.append(right_norm@robot_pos[:2])
                    constraints.append(right_norm@right_point)
                    print('right')
                    self.vectors_doors.append(center-right_point)
                    self.points_doors.append(right_point)

                    if robot_pos[1] < bot_point[1]:
                        print('right bot')
                        robot_norms.append(bot_norm@robot_pos[:2])
                        constraints.append(bot_norm@bot_point)
                        self.vectors_doors.append(center-bot_point)
                        self.points_doors.append(bot_point)
                    elif robot_pos[1] > top_point[1]:
                        print('right top')
                        robot_norms.append(top_norm@robot_pos[:2])
                        constraints.append(top_norm@top_point)
                        self.vectors_doors.append(center-top_point)
                        self.points_doors.append(top_point)
                
                elif robot_pos[1] < bot_point[1]: # bottom side of obstacle
                    robot_norms.append(bot_norm@robot_pos[:2])
                    constraints.append(bot_norm@bot_point)
                    print('bot')
                    self.vectors_doors.append(center-bot_point)
                    self.points_doors.append(bot_point)

                    if robot_pos[0] < left_point[0]:
                        print('bot left')
                        robot_norms.append(left_norm@robot_pos[:2])
                        constraints.append(left_norm@left_point)
                        self.vectors_doors.append(center-left_point)
                        self.points_doors.append(left_point)
                    elif robot_pos[0] > right_point[0]:
                        print('bot right')
                        robot_norms.append(right_norm@robot_pos[:2])
                        constraints.append(right_norm@right_point)
                        self.vectors_doors.append(center-right_point)
                        self.points_doors.append(right_point)
                
                elif robot_pos[1] > top_point[1]:
                    print('top')
                    robot_norms.append(top_norm@robot_pos[:2])
                    constraints.append(top_norm@top_point)
                    self.vectors_doors.append(center-top_point)
                    self.points_doors.append(top_point)

                    if robot_pos[0] < left_point[0]:
                        print('top left')
                        robot_norms.append(left_norm@robot_pos[:2])
                        constraints.append(left_norm@left_point)
                        self.vectors_doors.append(center-left_point)
                        self.points_doors.append(left_point)
                    elif robot_pos[0] > right_point[0]:
                        print('top right')
                        robot_norms.append(right_norm@robot_pos[:2])
                        constraints.append(right_norm@right_point)
                        self.vectors_doors.append(center-right_point)
                        self.points_doors.append(right_point)

        self.robot_norms = np.array(np.abs(robot_norms))
        self.constraints = np.array(np.abs(constraints)-r)
        self.robot_pos = [robot_pos[0], robot_pos[1]]
        self.vectors_walls = np.array(self.vectors_walls)/1
        self.vectors_doors = np.array(self.vectors_doors)/1
        self.points_walls = np.array(self.points_walls)
        self.points_doors = np.array(self.points_doors)
        self.normals = np.array(self.normals)

        # TODO:
            # Also output normal vectors
            # Add furnitures

        return self.robot_norms, self.constraints, self.normals

    def display(self) -> None:
        print('plotting')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(self.robot_pos[0], self.robot_pos[1], s=10, c='red')
        for point, vec in zip(self.points_walls, self.vectors_walls):
            ax.arrow(point[0], point[1], vec[0], vec[1])
            ax.scatter(point[0], point[1], s = 10, c = 'green')
        
        for point, vec in zip(self.points_doors, self.vectors_doors):
            ax.arrow(point[0], point[1], vec[0], vec[1], color='red')
            ax.scatter(point[0], point[1], s = 10, c = 'green')

        ax.set_ylim([-4.5, 4.5])
        ax.set_xlim([-9, 6.5])
        plt.show()