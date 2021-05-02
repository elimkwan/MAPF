from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as constant

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class Voronoii:
    def __init__(self, graph):
        self.G = graph #networkx object

    def next(self, node, t):
        # returns a list of (next_node, cost) tuples. this represents the children of node at time t.
        x = []

        for neighbor in self.G.neighbors(node):
            d = self.G.edges[node,neighbor]['distance']
            c = self.G.edges[node,neighbor]['capacity']

            cost = d * (1/(c+0.001)) * (t+1) # Both (Capacity, Distance)
            # cost = d * (t+1) + 0.001*1/(c+0.001)# Distance only
            # cost = 1/(c+0.001) + 0.001*d*(t+1)# Capacity Only
            x.append([neighbor, cost])
        return x

    def estimate(self, node, goal, t):
        # returns an estimate of the cost to travel from the current node to the goal node
        p1 = self.G.nodes[node]['position']
        p2 = self.G.nodes[goal]['position']
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return (np.linalg.norm(np.array([dx,dy]))+0.0001)

    def getEdgeCapacity(self, prev_node, node):
        c = self.G.edges[prev_node,node]['capacity']
        return c
        
    def getNodeCapacity(self, node):
        c = self.G.edges[node,node]['capacity']
        return c

    def getTotalDistanceAndArea(self):
        total_area = 0
        total_dist = 0
        assigned = {}
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    d = self.G.edges[n,e]['distance']
                    c = self.G.edges[n,e]['capacity']
                    
                    # if ((p1+p2) > thres): # edge (directional) with too low use probability will be eliminated
                    total_area += d * c * (constant.ROBOT_RADIUS*2)
                    total_dist += d
                    assigned[frozenset((n, e))] = 1
                    
        return total_dist, total_area

    def getTravelledDistance(self, paths):
        d = 0
        for path in paths:
            for i in range(len(path)-1):
                d += self.G.edges[path[i],path[i+1]]['distance']

        return d

    def getCoverage(self, exp):
        total_area = 0
        total_dist = 0
        assigned = {}
        fig, ax = plt.subplots(figsize=(6,6))
        plt.xlim(0,34)
        plt.ylim(0,34)
        count = 0
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    p1 = self.G.nodes[n]['position']
                    p2 = self.G.nodes[e]['position']
                    d = self.G.edges[n,e]['distance']
                    c = self.G.edges[n,e]['capacity']
                    assigned[frozenset((n, e))] = 1
                    
                    adjustp1 = Point(p1.y, p1.x)
                    adjustp2 = Point(p2.y, p2.x)
                    
                    refpt1 = adjustp1 if adjustp1.y <= adjustp2.y else adjustp2
                    refpt2 = adjustp1 if adjustp1.y > adjustp2.y else adjustp2
                    
                    if refpt1.x >= refpt2.x:
                        theta_rot = np.pi - np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
                    else:
                        theta_rot = np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
                        
                    if theta_rot >= np.pi/2:
                        theta = theta_rot - np.pi/2
                    else:
                        theta = theta_rot + np.pi/2
                    
                    dy = -(c/2)*np.sin(theta)
                    if refpt1.y == refpt2.y:
                        dx = 0
                        width = d
                        height = c
                        a = 0
                    elif refpt1.x > refpt2.x:
                        dx = -(c/2)*np.cos(theta)
                        width = c
                        height = d
                        a = (theta) * 180 / np.pi
                    elif refpt1.x == refpt2.x:
                        dx = -(c/2)
                        width = c
                        height = d
                        a = 0
                    else:
                        dx = (c/2)*np.cos(np.pi - theta)
                        width = d
                        height = c
                        a = (theta_rot) * 180 / np.pi

                    rect = Rectangle((refpt1.x+dx,refpt1.y+dy),width,height,linewidth=0.1,fill=True,angle = a,color = 'black')
                    plt.gca().add_patch(rect)

        for o in exp.obstacles_loc:
            adjustedx, adjustedy = o[1],o[0]
            rect = Rectangle((adjustedx-0.5,adjustedy-0.5),1,1,linewidth=0.1,fill=True,angle = 0, color = 'black')
            plt.gca().add_patch(rect)
            
        ax.axis('off')

        im = fig
        im.canvas.draw()
        X = np.array(im.canvas.renderer._renderer)
        X_reshape = X.reshape((-1,4))
        X_reshape = np.delete(X_reshape, [1,2,3], axis = 1)
        black = np.count_nonzero(X_reshape == 0)
        white= np.count_nonzero(X_reshape == 255)
        print("Black px", black, "White px", white)
        percentage = black/(white+black)
        return percentage

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
                pos = self.G.nodes[paths[a][t]]['position']
                congestion[getSubGrid(pos.x),getSubGrid(pos.y)] += 1
            
            acc_congestion.append(congestion)
            congestion_flat = congestion.flatten()
            maxx.append(np.amax(congestion_flat))
            avgg.append(np.average(congestion_flat))

        maxx_maxx = np.amax(np.array(maxx))
        avgg_avgg = np.average(np.array(avgg))
        return acc_congestion, maxx_maxx, avgg_avgg
