from core.DataLoader import *
from core.DataStructure import OccupancyGrid
import core.Constant as constant

import numpy as np

class Experiment:
    def __init__(self, scene = None):
        self.initialised = False
        self.scene = scene

        self.start_locations = None
        self.end_locations = None
        self.start_end_pair = None
        self.occupancy_grid = None
        self.image = None

        #for Voronoi Diagram
        self.nodes = None
        self.edges_dir = None
        self.edges = None
        self.start_nodes = None
        self.end_nodes = None
        self.obstacles_loc = None

    def setParameters(self):
        self.obstacles_loc, self.image = rawInputToArr()
        resolution = 0.5
        self.occupancy_grid = OccupancyGrid(np.array(self.image), [0,0], resolution)
        start_locations, end_locations, self.start_end_pair= rawSceneToArr(self.scene)

        self.nodes, self.edges_dir, self.edges, self.start_nodes, self.end_nodes, skipped = getVoronoiiParam(
            obstacles_loc = self.obstacles_loc, 
            occupancy_grid = self.occupancy_grid, 
            start_end_pair = self.start_end_pair)

        start_locations_tmp = np.array(start_locations[:(constant.NUM_OF_AGENT+len(skipped))])
        end_locations_tmp = np.array(end_locations[:(constant.NUM_OF_AGENT+len(skipped))])
        self.start_locations = np.delete(start_locations_tmp, skipped, axis = 0)
        self.end_locations = np.delete(end_locations_tmp, skipped, axis = 0) 

        self.initialised = True
        print("Set Hyper Parameters, solving for", len(self.start_locations), "agents")

