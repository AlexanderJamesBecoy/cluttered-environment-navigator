import gym
from door import Door
import pybullet as p
import numpy as np
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from ObstacleConstraintGenerator import ObstacleConstraintsGenerator

import os

HEIGHT = 0.5
WIDTH = 0.3
SCALE = 1.3

class House:
# This class contains all the useful, but approximated, information that describes a house for which a mobile manipulator
# will explore. It will contain the goal objects which are the door knobs that is our mobile manipulator will try to reach.
# Further details of class House:
# - Walls
# - Doors
# - Furniture

    def __init__(self, env, robot_dim: list, scale: float):
        self._env = env
        self._offset = np.array([7.0, 3.5])
        self._points = {
            'A': np.array([0.0,0.0]),   # Wall vertex.
            'B': np.array([12.0,0.0]),  # Wall vertex.
            'C': np.array([0.0,7.0]),   # Wall vertex.
            'D': np.array([7.0,7.0]),   # Wall vertex.
            'E': np.array([8.0,7.0]),   # Wall vertex / Door hinge to outside.
            'F': np.array([11.0,7.0]),  # Wall vertex.
            'G': np.array([11.0,3.0]),  # Wall vertex.
            'H': np.array([12.0,3.0]),  # Wall vertex.
            'I': np.array([8.0,3.0]),   # Wall vertex / Door hinge to the kitchen.
            'J': np.array([5.0,7.0]),   # Wall vertex.
            'K': np.array([5.0,3.0]),   # Wall vertex.
            'L': np.array([6.0,7.0]),   # Wall vertex.
            'M': np.array([6.0,3.0]),   # Wall vertex.
            'N': np.array([8.0,4.0]),   # Wall vertex.
            'O': np.array([0.0,3.0]),   # Wall vertex.
            'P': np.array([3.0,3.0]),   # Wall vertex / Door hinges to the two bedrooms.
            'Q': np.array([3.0,0.0]),   # Wall vertex.
            'R': np.array([3.0,2.0]),   # Wall vertex.
            'S': np.array([10.0,3.0]),  # Wall vertex.
            'T': np.array([10.0,1.0]),  # Wall vertex.
            'U': np.array([11.0,1.0]),  # Wall vertex.
            'V': np.array([4.0,3.0]),   # Wall vertex.
            'W': np.array([12.0,1.0]),  # Wall vertex / Door hinge to the bathroom.
        }
        for x in self._points:  # Center the points around the origin.
            self._points[x] = (self._points[x] - self._offset)*SCALE

        self._doors = {}
        self._furniture = {}
        self.Obstacles = ObstacleConstraintsGenerator(robot_dim=robot_dim, scale=scale)

    def add_wall(self, start_pos, end_pos, wall_thickness=0.1, wall_height=0.5):
    # This function draws a wall segment into gym `env` from a starting position `start_pos` to a final position `end_pos`.
    # The default thickness `wall_thickness` and `wall_height` are 10 cm and 50 cm, respectively. They are modifiable.

        vec = end_pos - start_pos       # Obtain a vector from the two points.
        avg = (end_pos + start_pos)/2   # Obtain the average point between the two points, because
                                        # gym `env` draws the shape centered.
        theta = np.arctan2(*vec)        # Obtain the angle of the vector.
    
        dim = np.array([WIDTH, np.linalg.norm(vec), HEIGHT])    # Obtain the dimension of the wall.
        pos = [[avg[0], avg[1], theta]]                         # Describe the position of the wall with average position and angle.
        self.Obstacles.walls.append({'x': pos[0][0], 'y': pos[0][1], 'theta': pos[0][2], 'width': dim[0], 'length': dim[1], 'height': dim[2]}) # Add new obstacle pos to list
        self._env.add_shapes(shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos)

    def generate_walls(self):
    # This function generates and draw the fixed wall segments described in `self._points`.

        points = np.array([ # Generate wall edges
            [self._points['A'], self._points['B']],
            [self._points['A'], self._points['C']],
            [self._points['C'], self._points['D']],
            [self._points['E'], self._points['F']],
            [self._points['F'], self._points['G']],
            [self._points['B'], self._points['H']],
            [self._points['I'], self._points['H']],
            [self._points['J'], self._points['K']],
            [self._points['V'], self._points['M']],
            [self._points['L'], self._points['M']],
            [self._points['E'], self._points['N']],
            [self._points['O'], self._points['P']],
            [self._points['Q'], self._points['R']],
            [self._points['S'], self._points['T']],
            [self._points['T'], self._points['U']],
        ])

        for i in range(points.shape[0]):
        # Iterate for every wall edge, and draw the wall.
            start_pos = points[i][0]
            end_pos = points[i][1] 
            self.add_wall(start_pos, end_pos)
        
        self.Obstacles.walls = np.array(self.Obstacles.walls)

    def add_furniture(self, urdf, loc, pos_x, pos_y, pos_z=0.0):
    # This function adds a furniture to the `self._furniture` dictionary given the name and file location `loc` of the `urdf`,
    # and the 3D position of it in x-axis for `pos_x`, in y-axis for `pos_y`, and in z-axis for `pos_z`.
        urdf_loc = loc + '.urdf'
        urdfObstDict = {
            'type': 'urdf',
            'geometry': {'position': [pos_x, pos_y, pos_z]},
            'urdf': os.path.join(os.path.dirname(__file__), urdf_loc),
        }
        self._furniture[urdf] = UrdfObstacle(name=urdf, content_dict=urdfObstDict)
        # self.Obstacles[urdf].append(self._furniture[urdf]) # TODO

    def add_furniture_box(self, pos, dim):
    # This function creates a furniture in a shape of a cube. It will take the center position `pos`
    # and the dimension `dim`.
        self._env.add_shapes(
            shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=pos
        )

    def generate_furniture(self):
    # Add all the furnitures into the `self._furniture` dictionary.
        ### Bottom bedroom
        self.add_furniture(
            urdf='bottom_bedroom_bed_west',
            loc='resources/objects/cob_simulation/objects/bed_west', 
            pos_x=(self._points['Q'][0].item()-1.0*SCALE), 
            pos_y=(self._points['Q'][1].item()+0.5*SCALE),
        )
        self.add_furniture(
            urdf='bottom_bedroom_cabinet',
            loc='resources/objects/cob_simulation/objects/cabinet_ikea_malm_big', 
            pos_x=(self._points['A'][0].item()+0.7), 
            pos_y=(self._points['A'][1].item()+0.4),
        )
        self.add_furniture(
            urdf='bottom_bedroom_desk',
            loc='resources/objects/pr_assets/furniture/table', 
            pos_x=((2.0*self._points['O'][0].item()+1.5*SCALE)/2.0), 
            pos_y=(self._points['O'][1].item()-0.4*SCALE),
        )
        self.add_furniture(
            urdf='bottom_bedroom_chair',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
            pos_x=((2.0*self._points['O'][0].item()+1.5*SCALE)/2.0), 
            pos_y=(self._points['O'][1].item()-0.4*SCALE)-0.44,
        )
        ### Bottom bedroom

        ### Top bedroom
        self.add_furniture(
            urdf='top_bedroom_bed_north_1',
            loc='resources/objects/cob_simulation/objects/bed_north', 
            pos_x=((self._points['O'][0].item() + self._points['P'][0].item())/2.0 - 0.05), 
            pos_y=(self._points['Q'][1].item()/2.0 + 2.8)
        )
        self.add_furniture(
            urdf='top_bedroom_bed_north_2',
            loc='resources/objects/cob_simulation/objects/bed_north', 
            pos_x=((self._points['O'][0].item() + self._points['P'][0].item())/2.0 + 1.0), 
            pos_y=(self._points['Q'][1].item()/2.0 + 2.8)
        )
        self.add_furniture(
            urdf='top_bedroom_desk',
            loc='resources/objects/pr_assets/furniture/table', 
            pos_x=((2.0*self._points['C'][0].item()+1.5*SCALE)/2.0), 
            pos_y=(self._points['C'][1].item()-0.4*SCALE),
        )
        self.add_furniture(
            urdf='bottom_bedroom_chair',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
            pos_x=((2.0*self._points['C'][0].item()+1.5*SCALE)/2.0), 
            pos_y=(self._points['C'][1].item()-0.4*SCALE)-0.44,
        )
        self.add_furniture(
            urdf='top_bedroom_wardrobe',
            loc='resources/objects/cob_simulation/objects/cabinet_ikea_galant', 
            pos_x=(self._points['J'][0].item()-1.5*SCALE), 
            pos_y=(self._points['J'][1].item()-0.3*SCALE),
        )
        self.add_furniture(
            urdf='top_bedroom_cabinet',
            loc='resources/objects/cob_simulation/objects/cabinet_ikea_malm_big', 
            pos_x=(self._points['J'][0].item()-0.5*SCALE), 
            pos_y=(self._points['J'][1].item()-0.3*SCALE),
        )
        ### Top bedroom

        ### Living room
        self.add_furniture(
            urdf='living_room_table',
            loc='resources/objects/pr_assets/furniture/table', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
        )
        self.add_furniture(
            urdf='living_room_chair_1',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_south', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-0.35),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0+0.44),
        )
        self.add_furniture(
            urdf='living_room_chair_2',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_south', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+0.35),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0+0.44),
        )
        self.add_furniture(
            urdf='living_room_chair_3',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-0.35),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0-0.44),
        )
        self.add_furniture(
            urdf='living_room_chair_4',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+0.35),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0-0.44),
        )
        self.add_furniture(
            urdf='living_room_chair_5',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_east', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-1.0),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
        )
        self.add_furniture(
            urdf='living_room_chair_6',
            loc='resources/objects/cob_simulation/objects/chair_ikea_borje_west', 
            pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+1.0),
            pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
        )
        self.add_furniture(
            urdf='living_room_couch',
            loc='resources/objects/cob_simulation/objects/couch', 
            pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
            pos_y=((self._points['I'][1].item()-0.75)),
        )
        self.add_furniture(
            urdf='living_room_plant',
            loc='resources/objects/cob_simulation/objects/plant_floor_big', 
            pos_x=(self._points['S'][0].item()-0.6),
            pos_y=((self._points['S'][1].item()-0.6)),
        )
        self.add_furniture(
            urdf='living_room_tv_table',
            loc='resources/objects/cob_simulation/objects/table_tv', 
            pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
            pos_y=((self._points['A'][1].item() + 0.32*SCALE)),
        )
        self.add_furniture(
            urdf='living_room_tv',
            loc='resources/objects/cob_simulation/objects/tv_samsung', 
            pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
            pos_y=((self._points['A'][1].item() + 0.32*SCALE)),
            pos_z=(0.58)
        )
        self.add_furniture(
            urdf='living_room_bookcase_1',
            loc='resources/objects/cob_simulation/objects/cabinet_living_room_vertical', 
            pos_x=(self._points['L'][0].item()+0.5),
            pos_y=(self._points['L'][1].item()-1.0),
        )
        self.add_furniture(
            urdf='living_room_bookcase_2',
            loc='resources/objects/cob_simulation/objects/cabinet_living_room_vertical', 
            pos_x=(self._points['L'][0].item()+0.5),
            pos_y=(self._points['L'][1].item()-2.5),
        )
        ### Living room

        ### Kitchen     # Missing urdf for these furniture
        self.add_furniture_box( # Counter, sink # TODO: into objects?
            pos=[[(self._points['E'][0].item()+self._points['F'][0].item())/2.0,self._points['E'][1].item()-0.4,0]],
            dim=np.array([np.abs(self._points['E'][0].item()-self._points['F'][0].item())-0.3, 0.6, 0.9]),
        )
        self.add_furniture_box( # Counter, oven
            pos=[[self._points['F'][0].item()-0.4,(self._points['F'][1].item()+self._points['G'][1].item())/2.0-0.3,0]],
            dim=np.array([0.6, np.abs(self._points['F'][1].item()-self._points['G'][1].item())-0.6, 0.9]),
        )
        # self.add_furniture(
        #     urdf='kitchen_refridgerator',
        #     loc='resources/objects/cob_simulation/objects/refridgerator', 
        #     pos_x=(self._points['G'][0].item()-1.0),
        #     pos_y=(self._points['G'][1].item()+0.5),
        # )
        ### Kitchen

        ### Bathroom    # Missing urdf for these furniture
        self.add_furniture_box( # Bathtub
            pos=[[(self._points['S'][0].item()+self._points['H'][0].item())/2.0,self._points['S'][1].item()-0.45,0]],
            dim=np.array([np.abs(self._points['S'][0].item()-self._points['H'][0].item())-0.3, 0.8, 0.6]),
        )
        self.add_furniture_box( # Toilet, backside
            pos=[[self._points['T'][0].item()+0.3,self._points['T'][1].item()+1.2,0]],
            dim=np.array([0.3, 0.45, 0.8]),
        )
        self.add_furniture_box( # Toilet, seat
            pos=[[self._points['T'][0].item()+0.65,self._points['T'][1].item()+1.2,0]],
            dim=np.array([0.4, 0.45, 0.4]),
        )
        ### Bathroom

        for furniture in self._furniture:
        # Add all furniture in `self._furniture` dictionary into the Gym environment.
            self._env.add_obstacle(self._furniture[furniture])

    def add_door(self, room, pos, theta, is_open=True, is_flipped=False):
        # Add a door depending on its hinge 2D pose location `pos` and orientation `theta`, 
        # and append it into `self._doors` dictionary and Obstacles' `doors` list. 
        # Add options whether the door is open or mirrored.
        self._doors[room] = Door(self._env, pos=pos, is_open=is_open, theta=theta, is_flipped=is_flipped, scale=SCALE)
        self._doors[room].draw_door()
        self.Obstacles.doors.append({'x': self._doors[room].pos_door[0][0], 'y': self._doors[room].pos_door[0][1], 
                                        'theta': self._doors[room].pos_door[0][2], 'width': self._doors[room].dim_door[0], 
                                        'length': self._doors[room].dim_door[1], 'height': self._doors[room].dim_door[2]})
        self.Obstacles.knobs.append({'x': self._doors['bathroom'].pos_knob[0][0], 'y': self._doors['bathroom'].pos_knob[0][1], 
                                        'theta': self._doors[room].pos_knob[0][2], 'width': self._doors[room].dim_knob[0], 
                                        'length': self._doors[room].dim_knob[1], 'height': self._doors[room].dim_knob[2]})

    def generate_doors(self):
        # Add all door and door knobs to pos list and convert all lists to np arrays
        self.add_door(room='bathroom', pos=self._points['W'], theta=0.0, is_open=True, is_flipped=True)
        self.add_door(room='outdoor', pos=self._points['E'], theta=np.pi, is_open=False)
        self.add_door(room='top_bedroom', pos=self._points['P'], theta=0.0, is_open=True)
        self.add_door(room='bottom_bedroom', pos=self._points['P'], theta=0.5*np.pi, is_open=True, is_flipped=True)
        self.add_door(room='kitchen', pos=self._points['I'], theta=-0.5*np.pi, is_open=True, is_flipped=True)
            
        # Append the door into list of Obstacles' `doors`.
        self.Obstacles.doors = np.array(self.Obstacles.doors)
        self.Obstacles.knobs = np.array(self.Obstacles.knobs)

