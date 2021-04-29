from core.DataStructure import OccupancyGrid, Point
import core.Constant as constant

import numpy as np

class Grid(OccupancyGrid):
    MOVEMENTS = np.array([[1,0],[0,1],[-1,0],[0,-1],[0,0]])

    def next(self, node, t):
        curr_pos = np.array(node)
        next_pos = curr_pos + Grid.MOVEMENTS

        mask = [~self.isOccupied(p) for p in next_pos]
        next_pos = next_pos[mask]
        return list(zip(map(tuple,next_pos), np.ones(next_pos.shape[0])))


    def estimate(self, node1, node2, t):
        return abs(node1[0]-node2[0]) + abs(node1[1]-node2[1])

    def getOptimiserCost(self, paths=None, end_locations=None, total_free_space=1):
        dist_travelled = 0
        for a, path in enumerate(paths):
            for t, pos in enumerate(path):
                if t == 0 :
                    continue
                if np.all(pos == end_locations[a]):
                    continue
                dist_travelled += 1
        
        return dist_travelled/total_free_space

    def getCongestionLv(self, paths=None):
        def getSubGrid(loc):
            if loc == 0:
                return 0
            elif loc > 31:
                return -1
            else:
                return int((loc-1)/constant.REGION)
        
        sz = int(constant.MAP_SIZE/constant.REGION)
        acc_congestion = []
        total_time = np.array(paths).shape[1]
        agents = np.array(paths).shape[0]
        avgg = []
        maxx = []
        for t in range(total_time):
            congestion = np.zeros((sz,sz))
            for a in range(agents):
                pos = paths[a][t]
                congestion[getSubGrid(pos[0]),getSubGrid(pos[1])] += 1
            
            acc_congestion.append(congestion)
            congestion_flat = congestion.flatten()
            maxx.append(np.amax(congestion_flat))
            avgg.append(np.average(congestion_flat))

        maxx_maxx = np.amax(np.array(maxx))
        avgg_avgg = np.average(np.array(avgg))
        return acc_congestion, maxx_maxx, avgg_avgg



            


