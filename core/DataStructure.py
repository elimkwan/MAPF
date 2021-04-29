import core.Constant as c

import numpy as np
import scipy, scipy.signal

# class OccupancyGrid:
#     def __init__(self, values, origin, resolution):
#         self._original_values = values.copy()
#         self._values = values.copy()
#         # Inflate obstacles (using a convolution).
#         self.inflated_grid = np.zeros_like(values)
#         self.inflated_grid[values == c.OCCUPIED] = 1.
#         w = int(2 * c.ROBOT_RADIUS/ resolution)
#         self.inflated_grid = scipy.signal.convolve2d(self.inflated_grid, np.ones((w, w)), mode='same')
#         self._values[self.inflated_grid > 0.] = c.OCCUPIED
#         self._origin = np.array(origin[:2], dtype=np.float32)
#         self._origin -= resolution / 2.
#         self._resolution = resolution
    
#     def get_index(self, position):
#         idx = ((position - self._origin) / self._resolution).astype(np.int32)
#         if len(idx.shape) == 2:
#             idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
#             idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
#             return (idx[:, 0], idx[:, 1])
#         idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
#         idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
#         return tuple(idx)

#     def get_position(self, i, j):
#         return np.array([i, j], dtype=np.float32) * self._resolution + self._origin
    
#     def getRows(self):
#         return self._values.shape[0]
    
#     def getCols(self):
#         return self._values.shape[1]
    
#     def isOccupied(self, position):
#         shape = np.array(self._values).shape
#         #shape of (34,34), 33.5 is in bound
#         if position[0] >= (shape[0]) or \
#             position[1] >= (shape[1]) or \
#             position[0] < 0 or \
#             position[1] < 0:
#             return True
#         idx = self.get_index(position)
#         return self._values[idx] == c.OCCUPIED

#     def isFree(self, p):
#         print(p)
#         return not self.isOccupied([p.x,p.y])
    
#     def isValidLine(self,p1,p2):
#         d = 20 #descretization 
#         points = np.linspace([p1.x, p1.y], [p2.x,p2.y], d)
#         for p in points:
#             if self.isOccupied(p):
#                 return False
#         return True

#     def getArea(self):
#         #assume resolution is 1
#         return np.count_nonzero(np.array(self.inflated_grid).flatten() == c.FREE)

class OccupancyGrid:
    def __init__(self, values, origin, resolution):
        original_values = np.array(values)
        self.original_values = original_values
        org_inflated_grid = np.zeros_like(original_values)
        org_inflated_grid[original_values == c.OCCUPIED] = c.OCCUPIED

        scale = 1/resolution
        inflated_grid = org_inflated_grid.copy()
        inflated_grid = np.repeat(inflated_grid, scale, axis = 0)
        inflated_grid = np.repeat(inflated_grid, scale, axis = 1)
        self._values = inflated_grid

    def get_index(self, position):
        position = np.array(position)
        idx = ((position+0.25)/0.5).astype(np.int32)
        idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
        idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
        idx = tuple(idx)
        return tuple(idx)

    # def get_position(self, i, j):
    #     return np.array([i, j], dtype=np.float32) * self._resolution + self._origin
    
    def getRows(self):
        return self._values.shape[0]
    
    def getCols(self):
        return self._values.shape[1]
    
    def isOccupied(self, position):
        shape = np.array(self._values).shape
        #shape of (34,34), 33.5 is in bound
        if position[0] >= (shape[0]) or \
            position[1] >= (shape[1]) or \
            position[0] < 0 or \
            position[1] < 0:
            return True
        idx = self.get_index(position)
        return self._values[idx] == c.OCCUPIED

    def isFree(self, p):
        print(p)
        return not self.isOccupied([p.x,p.y])
    
    def isValidLine(self,p1,p2):
        d = 20 #descretization 
        points = np.linspace([p1.x, p1.y], [p2.x,p2.y], d)
        for p in points:
            if self.isOccupied(p):
                return False
        return True

    def getArea(self):
        #assume resolution is 1
        return np.count_nonzero(np.array(self.original_values).flatten() == c.FREE)



class Edge:
    def __init__(self, node_prev, node_next):
        self.prev = node_prev
        self.next = node_next
        self.edge_attr = {
            'distance':0.0, 
            'capacity':0.0, 
            'probability':0.0,
        }
    def setDistance(self, x):
        self.edge_attr['distance'] = x
        
    def setCapacity(self, x):
        self.edge_attr['capacity'] = x
    
    def setProbability(self, x):
        self.edge_attr['probability'] = x

class Point:
    def __init__ (self, x, y):
        self.x = x
        self.y = y