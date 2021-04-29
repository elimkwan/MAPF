from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as c

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
# import pandas as pd
import copy
import networkx as nx

def initEdgeCapacity(nodes, edges, occupancy_grid):
    
    def getSlope(p1,p2):
        if (p1.y == p2.y):
            return 100
        return (p1.x - p2.x) / ((p1.y - p2.y))
    
    def shiftLine(p1, p2, max_iter, left = False, expand_along_col = True):
        shifted_p1 = copy.deepcopy(p1)
        shifted_p2 = copy.deepcopy(p2)
        
        #Path width based on Robot Width
        robot_size = -1*(c.ROBOT_RADIUS*2) if left else 1*(c.ROBOT_RADIUS*2)

        for offset in range(1,int(max_iter/2)):
            if (expand_along_col):
                shifted_p1.y = p1.y + offset*robot_size
                shifted_p2.y = p2.y + offset*robot_size
            else:
                shifted_p1.x = p1.x + offset*robot_size
                shifted_p2.x = p2.x + offset*robot_size
            
            if not occupancy_grid.isValidLine(shifted_p1, shifted_p2):
                return (offset-1) if offset > 0 else 0
            
        return offset
    
    n = copy.deepcopy(nodes)
    for idx, edge in enumerate(edges):
        
        if (edge.prev == edge.next):
            continue
        
        p1 = copy.deepcopy(n[edge.prev])
        p2 = copy.deepcopy(n[edge.next])
        p1_l = copy.deepcopy(n[edge.prev])
        p2_l = copy.deepcopy(n[edge.next])

        m = abs(getSlope(p1,p2))
        max_iter = occupancy_grid.getRows() if m >= 1 else occupancy_grid.getCols()
        expand_col = True if m > 1 else False
        
        right_capacity = shiftLine(p1, p2, max_iter, expand_along_col = expand_col)
        left_capacity = shiftLine(p1_l, p2_l, max_iter, left = True, expand_along_col = expand_col)
        
        # total_capacity = left_capacity + right_capacity
        total_capacity = min(left_capacity,right_capacity)*2
        if total_capacity == 0:
            total_capacity = 1
        
        edge.setCapacity(total_capacity)

def initEdgeDistance(n, edges):
    for edge in edges:     
        dx = n[edge.prev].x - n[edge.next].x
        dy = n[edge.prev].y - n[edge.next].y
        dist = np.linalg.norm(np.array([dx,dy]))
        edge.setDistance(dist)

        if edge.prev == edge.next:
            edge.setProbability(0.1)
        else:
            edge.setProbability(0.5)
    return

def getVoronoiiGraph(occupancy_grid = None, nodes = None, edges = None, start_nodes = None, end_nodes = None):
    # # Create Voronoi Diagram
    # voronoi = Voronoi(obstacles_loc)

    # # Clean Voronoi Diagram
    # nodes_tmp, removed_nodes = removeInsidePoint(voronoi.vertices, occupancy_grid)
    # edges_tmp = removeInsideLine(len(voronoi.vertices), voronoi.ridge_vertices, removed_nodes)
    # edges_tmp = removeOtherInvalidLine(nodes_tmp, edges_tmp, occupancy_grid)
    # nodes, edges = cleanNodesEdge(nodes_tmp, edges_tmp)
    # nodes, edges, start_nodes, end_nodes, skipped = addStartEndNode(occupancy_grid, start_end_pair, nodes, edges)

    # if (early_return):
    #     return nodes, edges, start_nodes, end_nodes, skipped

    # Initialise Edge Distance Attributes
    initEdgeDistance(nodes[:], edges)

    #Initialise Edge Capacity Attributes
    initEdgeCapacity(nodes[:], edges, occupancy_grid)

    G = nx.Graph()

    for e in edges:
        bad_edge = (e.edge_attr['distance'] == 0 and e.prev != e.next) or e.edge_attr['distance'] > 20
        if not (bad_edge):
            G.add_edge(e.prev, e.next)
            G.edges[e.prev, e.next]['distance'] = e.edge_attr['distance']
            G.edges[e.prev, e.next]['capacity'] = e.edge_attr['capacity']

    for e in G.edges:
        if G.edges[e]['distance'] < 1 and G.edges[e]['distance'] > 0:
            G.edges[e[0],e[1]]['capacity'] = 4
        
    for n in G.nodes:
        G.nodes[n]['position'] = nodes[n]
        cap = []
        for neighbor in G.neighbors(n):
            cap.append(G.edges[n, neighbor]['capacity'])
            
        G.edges[n, n]['capacity'] = max(cap)

    return G