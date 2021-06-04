import core.Constant as c

import numpy as np
import scipy, scipy.signal

from skimage.transform import resize
from skimage import img_as_bool

class OccupancyGrid:
    def __init__(self, values, origin, resolution):
        original_values = np.array(values)
        self.original_values = original_values
        # org_inflated_grid = np.zeros_like(original_values)
        # org_inflated_grid[original_values == c.OCCUPIED] = c.OCCUPIED
        self.resolution = resolution
        scale = (self.original_values.shape[0]/resolution, self.original_values.shape[1]/resolution)
        # inflated_grid = org_inflated_grid.copy()
        # inflated_grid = np.repeat(inflated_grid, scale, axis = 0)
        # inflated_grid = np.repeat(inflated_grid, scale, axis = 1)

        grid = np.array(original_values, dtype=np.bool)
        transformed = img_as_bool(resize(grid, scale))
        transformed = transformed.astype(int)

        w = 2 * int(c.ROBOT_RADIUS / resolution) + 1
        inflat = scipy.signal.convolve2d(transformed , np.ones((w, w)), mode='same')
        inflat[inflat > 0.] = c.OCCUPIED
        self._values = inflat

    def get_index(self, position):
        # print("position", position)
        position = np.array(position)
        idx = ((position+self.resolution/2)/self.resolution).astype(np.int32)
        # print("idx", idx)
        # print("self._values.shape[0]", self._values.shape[0])
        idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
        idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
        idx = tuple(idx)
        # print("idx", idx)
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
        return not self.isOccupied([p.x,p.y])
    
    def isValidLine(self,p1,p2, tolerance = 0):
        d = 20 #descretization 
        clashed = 0
        points = np.linspace([p1.x, p1.y], [p2.x,p2.y], d)
        for p in points:
            if self.isOccupied(p):
                # return False
                clashed += 1
            if clashed > tolerance:
                return False
        return True

    def getArea(self):
        #assume resolution is 1
        return np.count_nonzero(np.array(self.original_values).flatten() == c.FREE)
    
    def set_to_occuiped(self, cur_p1, cur_p2, shifted_p1, shifted_p2, mid_p1, mid_p2):
        # d1 = 4 #descretization along the width
        d2 = 40 #descretization along the height

        line_list = [[cur_p1, cur_p2], [mid_p1, mid_p2], [shifted_p1, shifted_p2]]
        for l in line_list:
            points = np.linspace([l[0].x,l[0].y],[l[1].x,l[1].y], d2)
            shape = np.array(self._values).shape
            for p in points:
                if p[0] >= (shape[0]) or \
                    p[1] >= (shape[1]) or \
                    p[0] < 0 or \
                    p[1] < 0:
                    continue
                idx = self.get_index(p)
                self._values[idx] = c.OCCUPIED

        # idx1 = self.get_index(np.array([shifted_p1.x,shifted_p1.y]))
        # idx2 = self.get_index(np.array([shifted_p2.x,shifted_p2.y]))
        # points = np.linspace([idx1[0], idx1[1]], [idx2[0], idx2[1]], d2)
        # for p in points:
        #     self._values[(p[0],p[1])] = c.OCCUPIED

        return True




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