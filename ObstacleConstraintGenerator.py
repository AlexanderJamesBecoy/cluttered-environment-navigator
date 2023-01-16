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
        self.constraints = []
        self.vectors = {}
        self.points = {}
        self.robot_dim = robot_dim*scale # to be used to construct constraints later
        self.constraints = []
        self.robot_pos = 0
        self.normals = []
        self.vertices = []
        self.sides = []
        self.surfaces = []
        self.vertices_open = []
        self.act = []

    def getNearestFaces(self, p0: np.ndarray) -> np.ndarray:
        """
            Finds the closest obstacle surfaces to consider. Output is a binary array, with 1 denoting the surface will be active.
            Only run this if constraints and vertices were generated.
        """
        # Compute distance from robot to all surfaces and vertices
        self.act = np.ones(len(self.constraints))

        # Get all sides that point away from the robot
        V = p0[:2] - self.surfaces[:, 0]
        n = np.array([normal@v >= 0 for normal, v in zip(self.normals, V)])
        inverse_n =  1 - n
        # print([normal@v for normal, v in zip(self.normals, V)])
        # offsets = []
        # for offset, b in zip(self.constraints, n):
        #     if b:
        #         offsets.append(offset)
        #     else:
        #         offsets.append(-offset)

        # distances_to_faces = np.abs(self.normals@p0[:2] + offsets)
        # distances_to_faces[n] = 1e6
        distances_to_faces = np.array([np.linalg.norm(p0[:2] - s) for s in self.surfaces[:, 0]])
        distances_to_faces[n] = 1e6
        sorted_idx = np.argsort(distances_to_faces)
            # print(distances_to_faces)
        # print(closest_idx)
        self.act[sorted_idx[:2]] = 0 # Set the closest indices to 0 to activate the corresponding constraint
        # print(self.act)
        return self.act

    def generateConstraintsCylinder(self) -> np.ndarray:
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
        self.normals = []
        self.vertices = []
        self.constraints = []
        self.sides = []

        self.generateConstraints(obstacles=self.walls, obstacles_name='walls')
        self.generateConstraints(obstacles=self.doors, obstacles_name='doors')
        self.generateConstraints(obstacles=self.furnitures, obstacles_name='furnitures')

        self.constraints = np.array(self.constraints)
        self.normals = np.array(self.normals).reshape(-1, 2)
        self.surfaces = np.array(self.surfaces).reshape(-1, 2, 2)

        return self.constraints, self.normals

    def computeNormalVector(self, p1: list[float, float], p2: list[float, float]) -> list[float, float]:
        """
            Returns the normal vector of the line defined by points p1 and p2.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return [dy, -dx], [-dy, dx] # [dy, -dx] -> left and top side of obstacle, [-dy, dx] -> right and lower side of the obstacle
    
    def getVertices(self):
        self.vertices = []
        self.computeVertices(obstacles=self.walls, obstacles_name='walls')
        self.computeVertices(obstacles=self.doors, obstacles_name='doors')
        self.computeVertices(obstacles=self.furnitures, obstacles_name='furnitures')
        self.vertices = np.array(self.vertices)
        self.vertices_open = self.vertices
        ceiling_height = 1
        floor_trt = [np.max(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), -0.1]
        floor_tlt = [np.min(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), -0.1]
        floor_blt = [np.min(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), -0.1]
        floor_brt = [np.max(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), -0.1]

        floor_trb = [np.max(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), 0]
        floor_tlb = [np.min(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), 0]
        floor_blb = [np.min(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), 0]
        floor_brb = [np.max(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), 0]

        ceiling_trt = [np.max(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), ceiling_height+0.1]
        ceiling_tlt = [np.min(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), ceiling_height+0.1]
        ceiling_blt = [np.min(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), ceiling_height+0.1]
        ceiling_brt = [np.max(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), ceiling_height+0.1]

        ceiling_trb = [np.max(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), ceiling_height]
        ceiling_tlb = [np.min(self.vertices[:, :, 0]), np.max(self.vertices[:, :, 1]), ceiling_height]
        ceiling_blb = [np.min(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), ceiling_height]
        ceiling_brb = [np.max(self.vertices[:, :, 0]), np.min(self.vertices[:, :, 1]), ceiling_height]

        floor_vertices = [floor_trb, floor_tlb, floor_blb, floor_brb, floor_trt, floor_tlt, floor_blt, floor_brt]
        ceiling_vertices = [ceiling_trb, ceiling_tlb, ceiling_blb, ceiling_brb, ceiling_trt, ceiling_tlt, ceiling_blt, ceiling_brt]
        self.vertices = list(self.vertices)
        self.vertices.append(floor_vertices)
        self.vertices.append(ceiling_vertices)
        self.vertices = np.array(self.vertices)
        return np.array(self.vertices)

    def computeVertices(self, obstacles, obstacles_name: str):
        # Compute all vertices
        for obstacle in obstacles:
            if obstacles_name=='furnitures' or np.abs(obstacle['theta']) != np.pi/2:
                # Compute the corner locations and center of each side
                tl = [obstacle['x'] - obstacle['width']/2, obstacle['y'] + obstacle['length']/2, 0]
                tr = [obstacle['x'] + obstacle['width']/2, obstacle['y'] + obstacle['length']/2, 0]
                br = [obstacle['x'] + obstacle['width']/2, obstacle['y'] - obstacle['length']/2, 0]
                bl = [obstacle['x'] - obstacle['width']/2, obstacle['y'] - obstacle['length']/2, 0]

                tlt = [obstacle['x'] - obstacle['width']/2, obstacle['y'] + obstacle['length']/2, obstacle['height']]
                trt = [obstacle['x'] + obstacle['width']/2, obstacle['y'] + obstacle['length']/2, obstacle['height']]
                brt = [obstacle['x'] + obstacle['width']/2, obstacle['y'] - obstacle['length']/2, obstacle['height']]
                blt = [obstacle['x'] - obstacle['width']/2, obstacle['y'] - obstacle['length']/2, obstacle['height']]
            else:
                tl = [obstacle['x'] - obstacle['length']/2, obstacle['y'] + obstacle['width']/2, 0]
                tr = [obstacle['x'] + obstacle['length']/2, obstacle['y'] + obstacle['width']/2, 0] 
                br = [obstacle['x'] + obstacle['length']/2, obstacle['y'] - obstacle['width']/2, 0] 
                bl = [obstacle['x'] - obstacle['length']/2, obstacle['y'] - obstacle['width']/2, 0] 

                tlt = [obstacle['x'] - obstacle['length']/2, obstacle['y'] + obstacle['width']/2, obstacle['height']]
                trt = [obstacle['x'] + obstacle['length']/2, obstacle['y'] + obstacle['width']/2, obstacle['height']]
                brt = [obstacle['x'] + obstacle['length']/2, obstacle['y'] - obstacle['width']/2, obstacle['height']]
                blt = [obstacle['x'] - obstacle['length']/2, obstacle['y'] - obstacle['width']/2, obstacle['height']]
            
            # Append vertices
            vertices = [tl, tr, br, bl, tlt, trt, brt, blt]
            self.vertices.append(vertices)

    def generateConstraints(self, obstacles, obstacles_name: str) -> np.ndarray:
        vectors = []
        points = []
        for obstacle in obstacles:
            # Set center of the obstacle
            center = np.array([obstacle['x'], obstacle['y']])

            # Check if obstacle is out of range, can be improved by checking each side but takes more time
            # obstacles were not rotated
            if obstacles_name=='furnitures' or np.abs(obstacle['theta']) != np.pi/2:
                # Compute the corner locations and center of each side
                tl = [obstacle['x'] - obstacle['width']/2, obstacle['y'] + obstacle['length']/2, 0]
                tr = [obstacle['x'] + obstacle['width']/2, obstacle['y'] + obstacle['length']/2, 0]
                br = [obstacle['x'] + obstacle['width']/2, obstacle['y'] - obstacle['length']/2, 0]
                bl = [obstacle['x'] - obstacle['width']/2, obstacle['y'] - obstacle['length']/2, 0]

                left_point = [center[0] - obstacle['width']/2, center[1]]
                top_point = [center[0], center[1] + obstacle['length']/2]
                right_point = [center[0] + obstacle['width']/2, center[1]]
                bot_point = [center[0], center[1] - obstacle['length']/2]
            else:
                tl = [obstacle['x'] - obstacle['length']/2, obstacle['y'] + obstacle['width']/2, 0]
                tr = [obstacle['x'] + obstacle['length']/2, obstacle['y'] + obstacle['width']/2, 0]
                br = [obstacle['x'] + obstacle['length']/2, obstacle['y'] - obstacle['width']/2, 0]
                bl = [obstacle['x'] - obstacle['length']/2, obstacle['y'] - obstacle['width']/2, 0]

                left_point = [center[0] - obstacle['length']/2, center[1]]
                top_point = [center[0], center[1] + obstacle['width']/2]
                right_point = [center[0] + obstacle['length']/2, center[1]]
                bot_point = [center[0], center[1] - obstacle['width']/2]

            # Compute the normal vectors on each side
            left_norm = np.array(self.computeNormalVector(bl, tl)[0])
            top_norm = np.array(self.computeNormalVector(tl, tr)[0])
            right_norm = np.array(self.computeNormalVector(br, tr)[1])
            bot_norm = np.array(self.computeNormalVector(bl, br)[1])

            # Transform to unit vectors
            left_norm = left_norm / np.linalg.norm(left_norm)
            top_norm = top_norm / np.linalg.norm(top_norm)
            right_norm = right_norm / np.linalg.norm(right_norm)
            bot_norm = bot_norm / np.linalg.norm(bot_norm)

            self.normals = np.append(self.normals, [left_norm, right_norm, top_norm, bot_norm])
            self.constraints = np.append(self.constraints, [left_norm@left_point, right_norm@right_point, top_norm@top_point, bot_norm@bot_point])
            self.surfaces.append([[left_point, left_norm], [right_point, right_norm], [top_point, top_norm], [bot_point, bot_norm]])
            vectors.append(center-left_point)
            vectors.append(center-right_point)
            vectors.append(center-top_point)
            vectors.append(center-bot_point)
            points.append(left_point)
            points.append(right_point)
            points.append(top_point)
            points.append(bot_point)

            # Check which constraints should be active, append those to the final lists
            # Constrain is active if the robot is on that side of the obstacle. If it's diagonal to the obstacle, then multiple constraints are active
                    
            self.vectors[obstacles_name] = vectors
            self.points[obstacles_name] = points        

    def display(self, pos) -> None:
        print('plotting')
        self.robot_pos = pos
        active_walls = self.surfaces[self.act == 0]
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(self.robot_pos[0], self.robot_pos[1], s=10, c='red')
        print(active_walls)

        if 'walls' in self.points.keys():
            for point, vec in zip(self.points['walls'], self.vectors['walls']):
                ax.arrow(point[0], point[1], vec[0], vec[1])
                ax.scatter(point[0], point[1], s = 10, c = 'green')

        
        if 'doors' in self.points.keys():
            for point, vec in zip(self.points['doors'], self.vectors['doors']):
                ax.arrow(point[0], point[1], vec[0], vec[1], color='yellow')
                ax.scatter(point[0], point[1], s = 10, c = 'green')

        if 'furnitures' in self.points.keys():
            for point, vec in zip(self.points['furnitures'], self.vectors['furnitures']):
                ax.arrow(point[0], point[1], vec[0], vec[1], color='blue')
                ax.scatter(point[0], point[1], s = 10, c = 'green')
        
        for wall in active_walls:
            ax.scatter(wall[0][0], wall[0][1], s = 10, c= 'red')
            
        ax.set_ylim([-4.5, 4.5])
        ax.set_xlim([-9, 6.5])
        plt.show()

    def show_active_walls(self):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        plt.show()
# if robot_pos[0] < left_point[0]: # left side of obstacle
#         self.constraints.append(left_norm@left_point)
#         vectors.append(center-left_point)
#         points.append(left_point)
#         self.normals.append(left_norm)
#         self.sides.append('left')

#         # if robot_pos[1] < bot_point[1]:
#         #     self.constraints.append(bot_norm@bot_point)
#         #     vectors.append(center-bot_point)
#         #     points.append(bot_point)
#         #     self.normals.append(bot_norm)
#         #     self.sides.append('bot')
#         # elif robot_pos[1] > top_point[1]:
#         #     self.constraints.append(top_norm@top_point)
#         #     vectors.append(center-top_point)
#         #     points.append(top_point)
#         #     self.normals.append(top_norm)
#         #     self.sides.append('top')

#     elif robot_pos[0] > right_point[0]: # right side of obstacle
#         self.constraints.append(right_norm@right_point)
#         vectors.append(center-right_point)
#         points.append(right_point)
#         self.normals.append(right_norm)
#         self.sides.append('right')
        
#         # if robot_pos[1] < bot_point[1]:
#         #     self.constraints.append(bot_norm@bot_point)
#         #     vectors.append(center-bot_point)
#         #     points.append(bot_point)
#         #     self.normals.append(bot_norm)
#         #     self.sides.append('bot')
#         # elif robot_pos[1] > top_point[1]:
#         #     self.constraints.append(top_norm@top_point)
#         #     vectors.append(center-top_point)
#         #     points.append(top_point)
#         #     self.normals.append(top_norm)
#         #     self.sides.append('top')

#     elif robot_pos[1] < bot_point[1]: # bottom side of obstacle
#         self.constraints.append(bot_norm@bot_point)
#         vectors.append(center-bot_point)
#         points.append(bot_point)
#         self.normals.append(bot_norm)
#         self.sides.append('bot')

#         # if robot_pos[0] < left_point[0]:
#         #     self.constraints.append(left_norm@left_point)
#         #     vectors.append(center-left_point)
#         #     points.append(left_point)
#         #     self.normals.append(left_norm)
#         #     self.sides.append('left')
#         # elif robot_pos[0] > right_point[0]:
#         #     self.constraints.append(right_norm@right_point)
#         #     vectors.append(center-right_point)
#         #     points.append(right_point)
#         #     self.normals.append(right_norm)
#         #     self.sides.append('right')

#     elif robot_pos[1] > top_point[1]:
#         self.constraints.append(top_norm@top_point)
#         vectors.append(center-top_point)
#         points.append(top_point)
#         self.normals.append(top_norm)
#         self.sides.append('top')

#         # if robot_pos[0] < left_point[0]:
#         #     self.constraints.append(left_norm@left_point)
#         #     vectors.append(center-left_point)
#         #     points.append(left_point)
#         #     self.normals.append(left_norm)
#         #     self.sides.append('left')
#         # elif robot_pos[0] > right_point[0]:
#         #     self.constraints.append(right_norm@right_point)
#         #     vectors.append(center-right_point)
#         #     points.append(right_point)
#         #     self.normals.append(right_norm)
#         #     self.sides.append('right')