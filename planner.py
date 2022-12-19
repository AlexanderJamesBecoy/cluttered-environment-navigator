import numpy as np
from house import House

class Planner:

    def __init__(self, house):
        self._house = house

    def generate_initial_pos(self):
        pos_2d = np.array([
            (self._house._points['L'][0]+self._house._points['E'][0])/2.0, 
            (self._house._points['L'][1]+self._house._points['E'][1])/2.0 - 0.5,
            0.0,
        ])
        # return np.hstack((pos_2d, np.zeros(1)))
        return pos_2d

    def generate_waypoints(self):
        # points = self._house._points
        
        # waypoints = np.array([
            
        # ])
        pass

