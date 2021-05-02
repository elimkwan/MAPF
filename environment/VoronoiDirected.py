from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as constant

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class VoronoiDirected:
    def __init__(self, graph):
        self.G = graph #networkx object

    def next(self, node, t):
        # returns a list of (next_node, cost) tuples. this represents the children of node at time t.
        x = []

        for neighbor in self.G.neighbors(node):
            d = self.G.edges[node,neighbor,0]['distance']
            p = self.G.edges[node,neighbor,0]['probability'] #the direction factor
            c = self.G.edges[node,neighbor,0]['capacity']

            cost = d * (1/(p*c+0.001)) * (t+1) # Both (Capacity, Direction, Distance)
            # cost = d * (t+1) + 0.001*(1/(p*c+0.001))# Distance only
            # cost = 1/(p*c+0.001) + 0.001*d*(t+1) # Capacity and Direction Only
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
        c = self.G.edges[prev_node,node,0]['capacity']
        return c
    
    def getNodeCapacity(self, node):
        c = self.G.edges[node,node,0]['capacity']
        return c
    
    def getTotalDistance(self):
        thres = 0.1 #threshold for eliminating edge TODO: find ways to tune it systematically
        total_area = 0
        assigned = {}
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    d = self.G.edges[n,e,0]['distance']
                    c = self.G.edges[n,e,0]['capacity']
                    p1 = self.G.edges[n,e,0]['probability']
                    p2 = self.G.edges[e,n,0]['probability']
                    
                    # if ((p1+p2) > thres): # edge (directional) with too low use probability will be eliminated
                    total_area += d 
                    assigned[frozenset((n, e))] = 1
        return total_area
    
    def getOptimiserCost(self, solution, cost, start_nodes, end_nodes):
        #Form a Continuous Cost Function#
        if solution == None:
            last_nodes = np.array(start_nodes)
        else:
            last_nodes = np.array(solution)[:,-1]
        
        penality = (np.sum(np.linalg.norm(last_nodes-end_nodes))+1)
        print("penality", penality)
        # print("cbs cost", cost)
        # global_cost = (cost+1)*penality
        # if solution == None:
        #     print("Solution Not Found Cost", global_cost)
        # else:
        #     print("Normal Cost", global_cost)
        
        # print("\n")
            

        if solution == None:
            travelled_area = 0
            total_area = 1
            travelled_dist = constant.NUM_OF_AGENT
            num_of_agent = constant.NUM_OF_AGENT
        else:
            # given a set of paths, find the global cost
            total_area = self.getTotalDistance()
            travelled_area = 0
            travelled_dist = 0
            num_of_agent = len(solution)
            
            for agent_path in solution:
                agent_travelled_area = 0
                for idx in range(len(agent_path)-1):
                    cur, nextt = agent_path[idx], agent_path[idx+1]
                    
                    #find the transverse cost between cur and nextt
                    if (cur == nextt):
                        agent_travelled_area += 0
                    else:
                        d = self.G.edges[cur,nextt,0]['distance']
                        c = self.G.edges[cur,nextt,0]['capacity']
                        agent_travelled_area += d
                        travelled_dist += d
                travelled_area += agent_travelled_area

        ut = travelled_area/total_area
        cost_ut = 1/(ut+0.001)
        cost_ft = travelled_dist/num_of_agent
        global_cost = cost_ft * cost_ut * penality
        print("global cost", global_cost)
        # return [cost_ut*cost_ft], cost_ft, ut
        return global_cost, cost_ft, ut

    def formSubGraph(self, thres = 0.01, start_nodes = None, end_nodes = None):
        #threshold for eliminating edge TODO: find ways to tune it systematically
        total_area = 0
        assigned = {}
        subgraph_edge = []
        # print("Before edges", self.G.number_of_edges())
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    d = self.G.edges[n,e,0]['distance']
                    c = self.G.edges[n,e,0]['capacity']
                    p1 = self.G.edges[n,e,0]['probability']
                    p2 = self.G.edges[e,n,0]['probability']
                    
                    isImportant = n in start_nodes or e in start_nodes or n in end_nodes or e in end_nodes
                    if (isImportant or (1-p1-p2) > thres): # edge (directional) with too low use probability will be eliminated
                        # self.G.remove_edges(n,e,0)
                        # self.G.remove_edges(e,n,0)
                        subgraph_edge.append((n,e,0))
                        subgraph_edge.append((e,n,0))
                        subgraph_edge.append((n,n,0))
                        subgraph_edge.append((e,e,0))
                    # else:
                        # print("removed edges")
                    assigned[frozenset((n, e))] = 1

        self.G = self.G.edge_subgraph(subgraph_edge)
        # print("After edges", self.G.number_of_edges())
        return self.G

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
                    d = self.G.edges[n,e,0]['distance']
                    c = self.G.edges[n,e,0]['capacity']
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