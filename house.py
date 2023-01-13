import gym
import pybullet as p
import numpy as np
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from ObstacleConstraintGenerator import ObstacleConstraintsGenerator
import time
import os

HEIGHT = 2.0
WIDTH = 0.1
SCALE = 1.5
HEIGHT_KNOB = 1.0

DIMS = {
        'wall': {'width': 0.1, 'length': None, 'height': 1.5},
        'door': {'width': 0.2, 'length': None, 'height': 2.0, 'offset': 0.5},
        'knob': {'radius': 0.1, 'height': 1.0, 'offset': 0.3},
        'scale': 1.5,
}

class House:
    """
        This class contains all the useful, but approximated, information that describes a house for which a mobile manipulator
        will explore. It will contain the goal objects which are the door knobs that is our mobile manipulator will try to reach.
        Further details of class House:
         - Walls
         - Doors
         - Furniture
    """

    def __init__(self, env, robot_dim: list, scale: float, test_mode=False):
        self._env = env
        self.test_mode = test_mode
        # max_width = 0.0
        # max_length = 0.0
        if not test_mode:
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
                'X': np.array([10.0,0.0]),  # Living room bounding box
            }
            # max_width, max_length = self.set_offset()
            self._rooms = {
                'bathroom': [
                    {'x1': self._points['T'][0].item(), 'y1': self._points['T'][1].item(), 'x2': self._points['H'][0].item(), 'y2': self._points['H'][1].item()},
                ],
                'kitchen': [
                    {'x1': self._points['I'][0].item(), 'y1': self._points['I'][1].item(), 'x2': self._points['F'][0].item(), 'y2': self._points['F'][1].item()},
                ],
                'top_bedroom': [
                    {'x1': self._points['O'][0].item(), 'y1': self._points['O'][1].item(), 'x2': self._points['J'][0].item(), 'y2': self._points['J'][1].item()},
                ],
                'bottom_bedroom': [
                    {'x1': self._points['A'][0].item(), 'y1': self._points['A'][1].item(), 'x2': self._points['P'][0].item(), 'y2': self._points['P'][1].item()},
                ],
                'living_room': [
                    {'x1': self._points['H'][0].item(), 'y1': self._points['H'][1].item(), 'x2': self._points['E'][0].item(), 'y2': self._points['E'][1].item()},
                    {'x1': self._points['Q'][0].item(), 'y1': self._points['Q'][1].item(), 'x2': self._points['S'][0].item(), 'y2': self._points['S'][1].item()},
                    {'x1': self._points['X'][0].item(), 'y1': self._points['X'][1].item(), 'x2': self._points['W'][0].item(), 'y2': self._points['W'][1].item()},
                ],
            }
        else:
            self._offset = np.array([2.5, 2.5])
            self._points = {
                'A': np.array([0.0,0.0]),   # Wall vertex.
                'B': np.array([5.0,0.0]),   # Wall vertex.
                'C': np.array([0.0,5.0]),   # Wall vertex.
                'D': np.array([5.0,5.0]),   # Wall vertex.
            }
            # max_width, max_length = self.set_offset()
            self._rooms = {
                'test': [
                    {'x1': self._points['A'][0].item(), 'y1': self._points['A'][1].item(), 'x2': self._points['D'][0].item(), 'y2': self._points['D'][1].item()},
                ]
            }

        max_width = 0.0
        max_length = 0.0
        for x in self._points:  # Center the points around the origin.
            self._points[x] = (self._points[x] - self._offset)*SCALE
            max_width = self._points[x][0] if self._points[x][0] > max_width else max_width
            max_length = self._points[x][1] if self._points[x][1] > max_length else max_length

        self._corners = [(-1.0*self._offset*SCALE).tolist(), [max_width, max_length]]
        self._walls = []
        self._doors = {}
        self._furniture = []
        self.Obstacles = ObstacleConstraintsGenerator(robot_dim=robot_dim, scale=scale)
        self._test_mode = test_mode
        self._dims = DIMS

    def update(self, env):
        """
        Update a new Gym environment.
        """
        self._env = env

    def generate_walls(self):
        """
        Generate and draw the fixed wall segments described in `self._points`.
        """
        if not self._test_mode:
            self._wall_vertices = np.array([ # Generate wall edges
                [self._points['A'], self._points['Q']],
                [self._points['Q'], self._points['B']],
                [self._points['A'], self._points['O']],
                [self._points['O'], self._points['C']],
                [self._points['C'], self._points['J']],
                [self._points['J'], self._points['L']],
                [self._points['L'], self._points['D']],
                [self._points['E'], self._points['F']],
                [self._points['F'], self._points['G']],
                [self._points['B'], self._points['W']],
                [self._points['W'], self._points['H']],
                [self._points['I'], self._points['S']],
                [self._points['S'], self._points['G']],
                [self._points['G'], self._points['H']],
                [self._points['J'], self._points['K']],
                [self._points['V'], self._points['K']],
                [self._points['K'], self._points['M']],
                [self._points['L'], self._points['M']],
                [self._points['E'], self._points['N']],
                [self._points['O'], self._points['P']],
                [self._points['Q'], self._points['R']],
                [self._points['S'], self._points['T']],
                [self._points['T'], self._points['U']],
            ])
        else:
            self._wall_vertices = np.array([
                [self._points['A'], self._points['B']],
                [self._points['A'], self._points['C']],
                [self._points['B'], self._points['D']],
                [self._points['C'], self._points['D']],
            ])


        for wall_vertices in self._wall_vertices:
            # Iterate for every wall edge, and draw the wall.
            start_pos = wall_vertices[0]
            end_pos = wall_vertices[1] 
            vec = end_pos - start_pos       # Obtain a vector from the two points.
            avg = (end_pos + start_pos)/2   # Obtain the average point between the two points, because
                                            # gym `env` draws the shape centered.
            theta = np.arctan2(*vec)        # Obtain the angle of the vector.
        
            dim = np.array([self._dims['wall']['width'], np.linalg.norm(vec), self._dims['wall']['height']])    # Obtain the dimension of the wall.
            pos = [[avg[0], avg[1], theta]]                         # Describe the position of the wall with average position and angle.
            self._walls.append({'pos': pos, 'dim': dim})
            self.Obstacles.walls.append({'x': pos[0][0], 'y': pos[0][1], 'theta': pos[0][2], 'width': dim[0], 'length': dim[1], 'height': dim[2]}) # Add new obstacle pos to list
         
        self.Obstacles.walls = np.array(self.Obstacles.walls)

    def draw_walls(self):
        """
        Draw the walls into the Gym environment.
        """
        assert len(self._walls) > 0, f"There are no walls generated. Call house.generate_walls() before executing this method."

        for wall in self._walls:
            self._env.add_shapes(shape_type="GEOM_BOX", dim=wall['dim'], mass=0, poses_2d=wall['pos'])

    def add_furniture(self, urdf, loc, dim, pos_x, pos_y, pos_z=0.0):
        """
        Add a furniture to the `self._furniture` dictionary given the name and file location `loc` of the `urdf`,
        and the 3D position of it in x-axis for `pos_x`, in y-axis for `pos_y`, and in z-axis for `pos_z`.
        """
        urdf_loc = loc + '.urdf'
        urdfObstDict = {
            'type': 'urdf',
            'geometry': {'position': [pos_x, pos_y, pos_z]},
            'urdf': os.path.join(os.path.dirname(__file__), urdf_loc),
        }
        furniture = {
            'name': urdf,
            'urdf': UrdfObstacle(name=urdf, content_dict=urdfObstDict),
            'pos': [pos_x, pos_y, pos_z],
            'dim': dim,
        }
        self._furniture.append(furniture)
        self.Obstacles.furnitures.append({'x': pos_x, 'y': pos_y, 'width': dim[0], 'length': dim[1], 'height': dim[2]})
        # self.Obstacles[urdf].append(self._furniture[urdf]) # TODO

    def add_furniture_box(self, name, pos, dim):
        """
        Create a furniture in a shape of a cube. It will take the center position `pos`
        and the dimension `dim`.
        """
        furniture = {
            'name': name,
            'urdf': None,
            'pos': pos,
            'dim': dim,
        }
        self.Obstacles.furnitures.append({'x': pos[0], 'y': pos[1], 'width': dim[0], 'length': dim[1], 'height': dim[2]})
        self._furniture.append(furniture)

    def generate_furniture(self):
        """
        Add all the furnitures into the `self._furniture` dictionary.
        """
        if not self._test_mode:
            #########################################################################
            ### Bottom bedroom
            dim = [2.3,1.4,0.4]
            self.add_furniture(
                urdf='bottom_bedroom_bed_west',
                loc='resources/objects/cob_simulation/objects/bed_west', 
                pos_x=(self._points['Q'][0].item()-dim[0]/2.0), 
                pos_y=(self._points['Q'][1].item()+dim[1]/2.0),
                dim=dim,
            )
            dim = [1.4,0.8,0.7]
            self.add_furniture(
                urdf='bottom_bedroom_cabinet',
                loc='resources/objects/cob_simulation/objects/cabinet_ikea_malm_big', 
                pos_x=(self._points['A'][0].item()+dim[0]/2.0), 
                pos_y=(self._points['A'][1].item()+dim[1]/2.0),
                dim=dim,
            )
            dim = [2.4,1.1,0.7]
            self.add_furniture(
                urdf='bottom_bedroom_desk',
                loc='resources/objects/pr_assets/furniture/table', 
                pos_x=(self._points['O'][0].item()+dim[0]/2.0), 
                pos_y=(self._points['O'][1].item()-dim[1]/2.0),
                dim=dim,
            )
            dim = [0.6,0.8,1.0]
            self.add_furniture(
                urdf='bottom_bedroom_chair',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
                pos_x=self._points['O'][0].item()+1.0, 
                pos_y=self._points['O'][1].item()-0.5-0.6,
                dim=dim,
            )
            ### Bottom bedroom

            ### Top bedroom
            dim = [1.4,2.3,0.4]
            self.add_furniture(
                urdf='top_bedroom_bed_north_1',
                loc='resources/objects/cob_simulation/objects/bed_north', 
                pos_x=((self._points['O'][0].item() + self._points['P'][0].item())/2.0 - 0.05), 
                pos_y=(self._points['Q'][1].item()/2.0 + dim[1]*1.3),
                dim=dim,
            )
            self.add_furniture(
                urdf='top_bedroom_bed_north_2',
                loc='resources/objects/cob_simulation/objects/bed_north', 
                pos_x=((self._points['O'][0].item() + self._points['P'][0].item())/2.0 + 1.0), 
                pos_y=(self._points['Q'][1].item()/2.0 + dim[1]*1.3),
                dim=dim,
            )
            dim = [2.4,1.1,0.7]
            self.add_furniture(
                urdf='top_bedroom_desk',
                loc='resources/objects/pr_assets/furniture/table', 
                pos_x=self._points['C'][0].item()+dim[0]/2.0, 
                pos_y=self._points['C'][1].item()-dim[1]/2.0,
                dim=dim,
            )
            dim = [0.6,0.8,1.0]
            self.add_furniture(
                urdf='bottom_bedroom_chair',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
                pos_x=self._points['C'][0].item()+1.8/2.0, 
                pos_y=self._points['C'][1].item()-0.55-0.6,
                dim=dim,
            )
            dim = [1.8,0.7,0.8]
            self.add_furniture(
                urdf='top_bedroom_wardrobe',
                loc='resources/objects/cob_simulation/objects/cabinet_ikea_galant', 
                pos_x=(self._points['J'][0].item()-1.5*SCALE), 
                pos_y=(self._points['J'][1].item()-0.3*SCALE),
                dim=dim,
            )
            dim = [1.2,0.7,0.8]
            self.add_furniture(
                urdf='top_bedroom_cabinet',
                loc='resources/objects/cob_simulation/objects/cabinet_ikea_malm_big', 
                pos_x=(self._points['J'][0].item()-0.5*SCALE), 
                pos_y=(self._points['J'][1].item()-0.3*SCALE),
                dim=dim,
            )
            ### Top bedroom

            ### Living room
            dim = [2.4,1.1,0.7]
            self.add_furniture(
                urdf='living_room_table',
                loc='resources/objects/pr_assets/furniture/table', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
                dim=dim,
            )
            dim = [0.6,0.8,1.0]
            self.add_furniture(
                urdf='living_room_chair_1',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_south', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-0.35),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0+0.44),
                dim=dim,
            )
            self.add_furniture(
                urdf='living_room_chair_2',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_south', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+0.35),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0+0.44),
                dim=dim,
            )
            self.add_furniture(
                urdf='living_room_chair_3',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-0.35),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0-0.44),
                dim=dim,
            )
            self.add_furniture(
                urdf='living_room_chair_4',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_north', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+0.35),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0-0.44),
                dim=dim,
            )
            dim = [0.8,0.6,1.0]
            self.add_furniture(
                urdf='living_room_chair_5',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_east', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0-1.0),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
                dim=dim,
            )
            self.add_furniture(
                urdf='living_room_chair_6',
                loc='resources/objects/cob_simulation/objects/chair_ikea_borje_west', 
                pos_x=((self._points['P'][0].item()+self._points['M'][0].item())/2.0+SCALE*1.0+1.0),
                pos_y=((self._points['P'][1].item()+self._points['Q'][1].item())/2.0),
                dim=dim,
            )
            dim = [2.0,1.0,0.6]
            self.add_furniture(
                urdf='living_room_couch',
                loc='resources/objects/cob_simulation/objects/couch', 
                pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
                pos_y=((self._points['I'][1].item()-0.65)),
                dim=dim,
            )
            dim = [0.5,0.5,1.3]
            self.add_furniture(
                urdf='living_room_plant',
                loc='resources/objects/cob_simulation/objects/plant_floor_big', 
                pos_x=(self._points['S'][0].item()-0.4),
                pos_y=((self._points['S'][1].item()-0.5)),
                dim=dim,
            )
            dim = [1.8,0.64,0.7]
            self.add_furniture(
                urdf='living_room_tv_table',
                loc='resources/objects/cob_simulation/objects/table_tv', 
                pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
                pos_y=((self._points['A'][1].item() + 0.32*SCALE)),
                dim=dim,
            )
            dim = [1.8,0.20,0.6]
            self.add_furniture(
                urdf='living_room_tv',
                loc='resources/objects/cob_simulation/objects/tv_samsung', 
                pos_x=((self._points['I'][0].item()+self._points['S'][0].item())/2.0-SCALE*0.3),
                pos_y=((self._points['A'][1].item() + 0.32*SCALE)),
                pos_z=(0.58),
                dim=dim,
            )
            dim = [1.0,2.0,0.85]
            self.add_furniture(
                urdf='living_room_bookcase_1',
                loc='resources/objects/cob_simulation/objects/cabinet_living_room_vertical', 
                pos_x=(self._points['L'][0].item()+dim[0]/2.0),
                pos_y=(self._points['L'][1].item()-dim[1]/2.0),
                dim=dim,
            )
            self.add_furniture(
                urdf='living_room_bookcase_2',
                loc='resources/objects/cob_simulation/objects/cabinet_living_room_vertical', 
                pos_x=(self._points['L'][0].item()+dim[0]/2.0),
                pos_y=(self._points['L'][1].item()-dim[1]*(1.5)-0.1),
                dim=dim,
            )
            ### Living room

            ### Kitchen     # Missing urdf for these furniture
            self.add_furniture_box( # Counter, sink # TODO: into objects?
                name='kitchen_counter_sink',
                pos=[(self._points['E'][0].item()+self._points['F'][0].item())/2.0,self._points['E'][1].item()-0.4,0],
                dim=np.array([np.abs(self._points['E'][0].item()-self._points['F'][0].item())-0.3, 0.6, 0.9]),
            )
            self.add_furniture_box( # Counter, oven
                name='kitchen_counter_oven',
                pos=[self._points['F'][0].item()-0.4,(self._points['F'][1].item()+self._points['G'][1].item())/2.0-0.3,0],
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
                name='bathroom_bathtub',
                pos=[(self._points['S'][0].item()+self._points['H'][0].item())/2.0,self._points['S'][1].item()-0.45,0],
                dim=np.array([np.abs(self._points['S'][0].item()-self._points['H'][0].item())-0.3, 0.8, 0.6]),
            )
            self.add_furniture_box( # Toilet, backside
                name='bathroom_toilet_back',
                pos=[self._points['T'][0].item()+0.3,self._points['T'][1].item()+1.2,0],
                dim=np.array([0.3, 0.45, 0.8]),
            )
            self.add_furniture_box( # Toilet, seat
                name='bathroom_toilet_seat',
                pos=[self._points['T'][0].item()+0.65,self._points['T'][1].item()+1.2,0],
                dim=np.array([0.4, 0.45, 0.4]),
            )
            ### Bathroom
            ##############################################################################
        else:
            # @TEST_MODE:
            self.add_furniture_box(
                name='box_1',
                pos=[0.,0.,0.],
                dim=np.array([1.0,1.0,1.0])
            )

    def draw_furniture(self):
        """
        Draw the furniture into the Gym environment.
        """
        assert len(self._furniture) > 0, f"There are no furniture. Run house.generate_furniture() before executing this method."

        # ADD Furnitures ABOVE THIS LINE!
        for furniture in self._furniture:
        # Add all furniture in `self._furniture` dictionary into the Gym environment.
            # self._env.add_obstacle(self._furniture[furniture])
            if furniture['urdf'] is not None:
                self._env.add_obstacle(furniture['urdf'])
            else:
                self._env.add_shapes(shape_type="GEOM_BOX", dim=furniture['dim'], mass=0, poses_2d=[furniture['pos']])

    def add_door(self, room, pos, theta, is_flipped=False):
        """
        Add a door depending on its hinge 2D pose location `pos` and orientation `theta`, 
        and append it into `self._doors` dictionary and Obstacles' `doors` list. 
        Add options whether the door is open or mirrored.
        """
        self._doors[room] = Door(self._env, pos=pos, theta=theta, is_flipped=is_flipped, scale=SCALE)

    def generate_doors(self):
        """
        Add all door and door knobs to pos list and convert all lists to np arrays
        @param is_open - determines whether the door depending on the name is open or not.
        """
        assert self._test_mode is False, f"generate_doors() is is not accessible in test mode."

        self.add_door(room='bathroom', pos=self._points['W'], theta=0.0, is_flipped=True)
        self.add_door(room='outdoor', pos=self._points['E'], theta=np.pi)
        self.add_door(room='top_bedroom', pos=self._points['P'], theta=0.0)
        self.add_door(room='bottom_bedroom', pos=self._points['P'], theta=0.5*np.pi, is_flipped=True)
        self.add_door(room='kitchen', pos=self._points['I'], theta=-0.5*np.pi, is_flipped=True)

    def draw_doors(self, is_open):
        """
        Draw the doors.
        """
        assert self._test_mode is False

        self.Obstacles.doors = []
        self.Obstacles.knobs = []
        for room in self._doors:
            self._doors[room].draw_door(is_open[room])
            self.Obstacles.doors.append({'x': self._doors[room].pos_door[0][0], 'y': self._doors[room].pos_door[0][1], 
                                        'theta': self._doors[room].pos_door[0][2], 'width': self._doors[room].dim_door[0], 
                                        'length': self._doors[room].dim_door[1], 'height': self._doors[room].dim_door[2]})
            self.Obstacles.knobs.append({'x': self._doors['bathroom'].pos_knob[0][0], 'y': self._doors['bathroom'].pos_knob[0][1], 
                                        'theta': self._doors[room].pos_knob[0][2], 'width': self._doors[room].dim_knob[0], 
                                        'length': self._doors[room].dim_knob[1], 'height': self._doors[room].dim_knob[2]})
        # Append the door into list of Obstacles' `doors`.
        self.Obstacles.doors = np.array(self.Obstacles.doors)
        self.Obstacles.knobs = np.array(self.Obstacles.knobs)

    def get_room(self, x, y):
        """
        """
        x = x/SCALE + self._offset[0]
        y = y/SCALE + self._offset[1]
        for room in self._rooms:
            for box in self._rooms[room]:
                if (box['x1'] < x < box['x2']) and (box['y1'] < y < box['y2']):
                    return room
        return None

    def generate_plot_obstacles(self, door_generated=True):
        """
        Generate lines indicating walls, doors, door knobs and furniture for 2D plot.
        @return lists of coordinates describing lines (walls and doors), points (door knobs) and boxes (furniture).
        """
        # Declare a list storing all the lines and boxes.
        lines = []
        points = []
        boxes = []

        # Generate wall line coordinates
        for wall in self._wall_vertices:
            line = {
                'type': 'wall',
                'coord': wall.tolist()
            }
            lines.append(line)

        # Generate door line coordinates and knob point coordinates
            for room in self._doors:
                line = {
                    'type': 'door',
                    'coord': self._doors[room].get_line()
                }
                lines.append(line)

        if door_generated:
            for i in range(2):
                point = self._doors[room].knobs[i].get_pos()[0:2]
                points.append(point)

        # Generate furniture as box coordinates
        for furniture in self._furniture:
            box = {
                'x': furniture['pos'][0]-furniture['dim'][0]/2.0,
                'y': furniture['pos'][1]-furniture['dim'][1]/2.0,
                'w': furniture['dim'][0],
                'h': furniture['dim'][1],
            }
            boxes.append(box)
        
        return lines, points, boxes


class Door:
    """
    This class contains the useful information describing a door. It also contains the goal object `Knob`
    for which our mobile manipulator tries to reach.
    """
    
    def __init__(self, env, pos, theta, is_flipped=False, scale=1.0):
        """
        Initialize an object of this class. It requires the pointer to the gym environment `env`, the orientation of the door `theta`.
        Booleans describing whether the door is mirrored `is_flipped`, and whether the door is open `is_open` are set to False by default.
        """
        self.env = env
        self.pos = pos
        self.theta = theta
        self.scale = scale
        self.flipped = 1    # No mirroring of poses.
        self.open = 0       # No additive angle.
        self.dim_door = np.array([1.0*self.scale, DIMS['door']['width'], DIMS['door']['height']])
        self.dim_knob = np.array([DIMS['wall']['width']*self.scale, 0.3, 0.2]) # TODO -> into goal object
        self.knobs = []
        self.pos_door = []
        self.pos_knob = []

        if is_flipped:    # If the door is mirrored, -1 will mirror the required poses.
            self.flipped = -1

        knobs = []  # List of door knob objects.
    
    def draw_door(self, is_open):
        """
        Draw a door into gym `env`.
        """
        # If the door is open, an additive angle is added to rotate the door by additional 90 deg.
        if is_open:
            self.open = DIMS['door']['offset']*np.pi

        offset_x = DIMS['door']['offset']*self.scale*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y = DIMS['door']['offset']*self.scale*np.sin(self.theta+self.open*self.flipped)*self.flipped

        # Poses of 2D offset away from the center of the door to draw the doorknob. This is due to `env` drawing the shapes centered.
        offset_x_knob = DIMS['knob']['offset']*self.scale*np.cos(self.theta+self.open*self.flipped)*self.flipped
        offset_y_knob = DIMS['knob']['offset']*self.scale*np.sin(self.theta+self.open*self.flipped)*self.flipped
        
        # Absolute 2D poses describing the centered positions of the door and the doorknob, respectively.
        self.pos_door = [[self.pos[0]+offset_x, self.pos[1]+offset_y, self.theta+self.open*self.flipped]]
        self.pos_knob = [[self.pos[0]+offset_x+offset_x_knob, self.pos[1]+offset_y+offset_y_knob, self.theta+self.open*self.flipped]]

        # Draw the door and doorknob.
        self.env.add_shapes(shape_type="GEOM_BOX", dim=self.dim_door, mass=0, poses_2d=self.pos_door)
        # self.env.add_shapes(shape_type="GEOM_BOX",dim=self.dim_knob, mass=0, poses_2d=self.pos_knob)

        # Create door knob objects
        knob_offset_xy = DIMS['knob']['offset']/2.0
        knobs_offset = np.array([knob_offset_xy*self.scale*np.sin(self.theta+self.open*self.flipped), knob_offset_xy*np.cos(self.theta+self.open*self.flipped), 0])
        pos_knob = [
            np.hstack((np.array(self.pos_knob)[0][0:2], np.array([DIMS['knob']['height']]))) - knobs_offset,
            np.hstack((np.array(self.pos_knob)[0][0:2], np.array([DIMS['knob']['height']]))) + knobs_offset,
        ]
        for i in range(2):
            knob = Knob(self.env, pos_knob[i])
            knob.draw_knob()
            self.knobs.append(knob)

    def get_line(self):
        """
        Return the line coordinates describing the door on XY-plane.
        """
        x = self.pos[0] + self.dim_door[0]*np.cos(self.theta+self.open*self.flipped)*self.flipped
        y = self.pos[1] + self.dim_door[0]*np.sin(self.theta+self.open*self.flipped)*self.flipped
        return [self.pos.tolist(), [x,y]]

class Knob:
    """
    This class simulates a simple door knob. It only acts a static entity and is used as a goal for our mobile manipulator.
    """

    def __init__(self, env, pos_3d, radius=DIMS['knob']['radius']):
        """
        Declare public variables from the given arguments.
         - `pos_3d` describes the 3d pose of this knob.
         - `radius` describes the size of this knob; set default at 10 cm wide.
        """
        self.env = env
        self.pos = pos_3d
        self.radius = radius

    def draw_knob(self): # TODO -> into goal because add_shapes create obstacles
        """
        Draw the knob into the Gym environment `self.env`.
        """
        dim = np.array([self.radius])
        pos = [[self.pos[0], self.pos[1], 0]]
        self.env.add_shapes(shape_type="GEOM_SPHERE", dim=dim, mass=0, poses_2d=pos, place_height=self.pos[2])

    def get_pos(self):
        """
        Returns the 3D position of the door knob `self.pos`.
        """
        return self.pos
