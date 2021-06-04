from core.DataLoader import *
from core.DataStructure import OccupancyGrid
import core.Constant as constant

import numpy as np

class Experiment:
    def __init__(self, scenemap=None, scene = None, objective="Both", num_agent = 1):
        self.initialised = False
        self.scenemap = scenemap
        self.scene = scene
        self.objective = objective

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
        self.NUM_OF_AGENT = num_agent
        self.ROBOT_RADIUS = constant.ROBOT_RADIUS

    def setParameters(self, specified_start = None, specified_end = None):
        self.obstacles_loc, self.image = rawInputToArr(self.scenemap)
        resolution = constant.RESOLUTION
        self.occupancy_grid = OccupancyGrid(np.array(self.image), [0,0], resolution)

        if np.any(specified_start == None):
            start_locations, end_locations, self.start_end_pair= rawSceneToArr(scene=self.scene, num_agent=self.NUM_OF_AGENT)
        else:
            start_locations = specified_start
            end_locations = specified_end
            aggr = {}
            for a in range (len(start_locations)):
                aggr[Point(start_locations[a,0], start_locations[a,1])] = Point(end_locations[a,0], end_locations[a,1])
            self.start_end_pair = aggr

        # self.NUM_OF_AGENT = len(start_locations)



        self.nodes, self.edges_dir, self.edges, self.start_nodes, self.end_nodes, skipped = getVoronoiiParam(
            obstacles_loc = self.obstacles_loc, 
            occupancy_grid = self.occupancy_grid, 
            start_end_pair = self.start_end_pair,
            num_agent = self.NUM_OF_AGENT)

        start_locations_tmp = np.array(start_locations[:(self.NUM_OF_AGENT+len(skipped))])
        end_locations_tmp = np.array(end_locations[:(self.NUM_OF_AGENT+len(skipped))])
        self.start_locations = np.delete(start_locations_tmp, skipped, axis = 0)
        self.end_locations = np.delete(end_locations_tmp, skipped, axis = 0) 

        # self.start_locations = np.array([[self.nodes[n].x,self.nodes[n].y] for n in self.start_nodes])
        # self.start_locations = np.round(self.start_locations).astype(int)
        # self.end_locations = np.array([[self.nodes[n].x,self.nodes[n].y] for n in self.end_nodes])
        # self.end_locations = np.round(self.end_locations).astype(int)

        self.initialised = True
        print("Set Hyper Parameters, solving for", len(self.start_locations), "agents")


    def getWaiting(self, paths=None, grid = False):
        # Matrice for congestion
        # Share among all environment parameterisation scheme
        if (np.all(paths == None) or np.array(paths).size < self.NUM_OF_AGENT) and not grid:
            paths = np.array(self.start_nodes).reshape((self.NUM_OF_AGENT,1))
        elif (np.array(paths).shape == (self.NUM_OF_AGENT,) and paths == [None]*self.NUM_OF_AGENT):
            #[None, None, None, None, None, None, None, None, None, None, None, None]
            paths = np.array(self.start_nodes).reshape((self.NUM_OF_AGENT,1))
        elif np.any(paths == None) and not grid:
            try:
                for idx, s in enumerate(paths):
                    if s == None:
                        paths[idx] = [self.start_nodes[idx]]
            except:
                print("Caught exceptions in getWaiting")
                paths = np.array(self.start_nodes).reshape((12,1))


        if grid:
            end = [(elem[0], elem[1])for elem in self.end_locations]
            count = 0
            for a, path in enumerate(paths):
                last_step = None
                for t, step in enumerate(path):
                    initialised = not np.any(last_step == None)
                    not_destination = (step[0], step[1]) not in end
                    if initialised and not_destination and step == last_step: 
                        count += 1
                    last_step = step
            count /= self.NUM_OF_AGENT     
            return count
        
        count = 0
        # print("paths", paths)
        for a, path in enumerate(paths):
            last_step = None
            for t, step in enumerate(path):
                initialised = not np.any(last_step == None)
                not_destination = step not in self.end_locations
                if step not in self.end_nodes:
                    count += 1
                last_step = step
        count /= self.NUM_OF_AGENT     
        return count

