from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as constant

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import re
import pandas as pd
import copy

def rawInputToArr():
#     f = open("./maze-32-32-4/maze-32-32-4.map", "r")
    f = open("./input/random-32-32-10/random-32-32-10.map", "r")
    lines = f.readlines()
    f.close()
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    img = np.zeros((height,width))
    obstacles = []
    for index in range (4, 4+height):
        cur_line = bytes(lines[index],'utf-8')
        k = np.array(list(cur_line), dtype = np.unicode)
        k = k[:width]
        
        y = np.array(np.where(k == '64'))
        y = y.reshape((y.shape[1]))
        x = np.tile((index-4), y.shape[0])
        
        pairs = np.stack((x,y), axis = 1)
        obstacles.extend(pairs)
    
    for p in obstacles:
        img[p[0],p[1]] = 1
        
    obstacles = [[x+1.5,y+1.5] for x,y in obstacles] #offset by 1.5 instead of 0.5, will add extra row and column at the begining
    
    #add boundary else Voronoi wont work
    img = np.array(img)
    img = np.insert(img, 0, 1, axis=0)
    img = np.insert(img, img.shape[0], 1, axis=0)
    img = np.insert(img, 0, 1, axis=1)
    img = np.insert(img, img.shape[1], 1, axis=1)
    # print(img.shape)
    
    max_i = img.shape[0]-1 #33
    s1 = np.arange(0.5, max_i+1, 0.5).round(1)
    s2 = np.tile(0.5, s1.shape[0])
    first_column = np.stack((s1,s2), axis = 1)
    first_column = np.delete(first_column, (0), axis=0)
    first_column = np.delete(first_column, (-1), axis=0)

    #add boundary to obstacle list
    s3 = np.tile(max_i+0.5 , s1.shape[0])
    last_column = np.stack((s1,s3), axis = 1)
    last_column = np.delete(last_column, (0), axis=0)
    last_column = np.delete(last_column, (-1), axis=0)

    first_row = np.stack((s2,s1), axis = 1)
    last_row = np.stack((s3,s1), axis = 1)
    
    final_obs = np.concatenate((obstacles, first_column, last_column, first_row, last_row))
#     final_obs =[Point(x,y) for x,y in final_obs]
    
    return final_obs, img

def rawSceneToArr(scene = "./input/random-32-32-10/scen-even/random-32-32-10-even-1.scen"):
#     f = open("./maze-32-32-4/scen-random/maze-32-32-4-random-1.scen", "r")
    f = open(scene, "r")
    lines = f.readlines()
    f.close()
    
    starts = []
    ends = []
    pairs = {}
    number_of_agent = constant.NUM_OF_AGENT + 10
    c = 0
    
    for index, l in enumerate(lines):
        x = l.split()
        match = re.match(r"random-32-32-10", x[1])
        if match:
            coord = np.array(x[4:8]).astype('float') + 1
            starts.append([coord[0], coord[1]])
            ends.append([coord[2], coord[3]])
            pairs[Point(coord[0], coord[1])] = Point(coord[2], coord[3])
            
            if c == number_of_agent:
                return starts, ends, pairs
            else:
                c += 1


def removeInsidePoint(v, occupancy_grid):
    new_v = []
    removed = []
    for index, p in enumerate (v):
        inside = occupancy_grid.isOccupied(p)
        if (inside):
            removed.append(index)
            new_v.append(Point(-10,-10))
            continue

        new_v.append(Point(p[0],p[1]))
    return new_v, removed

def removeInsideLine(length, ridge_vertices, removed_v):
    total_len = length
    new_ridge = []
    for p in ridge_vertices:
        p0 = total_len - abs(p[0]) if p[0] < 0 else p[0]
        p1 = total_len - abs(p[1]) if p[1] < 0 else p[1]
        if p0 in removed_v or p1 in removed_v:
            continue
        new_ridge.append([p0,p1])
    return new_ridge

def removeOtherInvalidLine(nodes, edges, occupancy_grid):
    new_edges = []
    for i, e in enumerate (edges):
        if occupancy_grid.isValidLine(nodes[e[0]], nodes[e[1]]):
            new_edges.append(e)
    return new_edges         

def cleanNodesEdge(prev_nodes, prev_edges):
#     print(prev_edges)
    m = {}
    j = 0
    new_nodes = []
    
    for i, node in enumerate(prev_nodes):
        if node == Point(-10,-10):
            continue
        m[i] = j #key is previous index, val is desired index
        new_nodes.append(node)
        j += 1
    
    #form list of Edge Object
    new_edges_dir = []
    new_edges = []
    for p, n in prev_edges:
#         print(m[p],m[n])
        new_edges_dir.append(Edge(m[p], m[n]))
        new_edges_dir.append(Edge(m[n], m[p]))
        new_edges_dir.append(Edge(m[p], m[p]))
        new_edges_dir.append(Edge(m[n], m[n]))

        new_edges.append(Edge(m[p], m[n]))
        new_edges.append(Edge(m[p], m[p]))
        new_edges.append(Edge(m[n], m[n]))
    
    return new_nodes, new_edges_dir, new_edges

def addStartEndNode(occupancy_grid, pairs, nodes, edges_dir, edges):
    ZeroDist = 0
    SomeDist = 1
    NotPossible = 2
    
    def appendNode(cur, nodes):
        distance = np.zeros((len(nodes),2))
        for i, n in enumerate (nodes):
            distance[i] = [np.linalg.norm(np.array([cur.x, cur.y]) - np.array([n.x, n.y])), int(i)]
        
        df = pd.DataFrame(distance)
        df.sort_values(by=0, ascending=True, inplace=True)
        
        for n in df[1]:
            n = int(n)
            dist = df.loc[n, 0]
            if dist == 0.0:
                return ZeroDist, n
            if occupancy_grid.isValidLine(cur, nodes[int(n)]):
                return  SomeDist, n
        return NotPossible, None
    
    def appendNode2(case, c_n, n):
        if case == SomeDist:
            m = len(nodes) + len(new_nodes)
            new_nodes.append(n)
            new_edges_dir.append(Edge(c_n, m))
            new_edges_dir.append(Edge(m, c_n))
            new_edges_dir.append(Edge(c_n, c_n))
            new_edges_dir.append(Edge(m, m))

            new_edges.append(Edge(c_n, m))
            new_edges.append(Edge(c_n, c_n))
            new_edges.append(Edge(m, m))
            
    
    new_nodes, new_edges_dir, new_edges = [], [], []
    start_nodes, end_nodes = [], []
    count = 0
    skipped = []
    for idx, start in enumerate(pairs):
        case1, connected_node1 = appendNode(start, nodes)
        case2, connected_node2 = appendNode(pairs[start], nodes)
        # print("index", idx, case1, case2, connected_node1, connected_node2)
        
        if (case1 == NotPossible or case2 == NotPossible):
            skipped.append(idx)
            continue
        
        count += 1
        appendNode2(case1, connected_node1, start)
        appendNode2(case2, connected_node2, pairs[start])
        start_nodes.append(connected_node1)
        end_nodes.append(connected_node2)

        if count == constant.NUM_OF_AGENT:
            break
    
    # for k in new_nodes:
    #     print("new nodes", k.x, k.y)
    nodes.extend(new_nodes)   
    edges_dir.extend(new_edges_dir)
    edges.extend(new_edges)
    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped



def getVoronoiiParam(obstacles_loc = None, occupancy_grid = None, start_end_pair = None):
    # Create Voronoi Diagram
    voronoi = Voronoi(obstacles_loc)

    # Clean Voronoi Diagram
    nodes_tmp, removed_nodes = removeInsidePoint(voronoi.vertices, occupancy_grid)
    edges_tmp = removeInsideLine(len(voronoi.vertices), voronoi.ridge_vertices, removed_nodes)
    edges_tmp = removeOtherInvalidLine(nodes_tmp, edges_tmp, occupancy_grid)

    nodes, edges_dir, edges = cleanNodesEdge(nodes_tmp, edges_tmp)
    nodes, edges_dir, edges, start_nodes, end_nodes, skipped = addStartEndNode(occupancy_grid, start_end_pair, nodes, edges_dir, edges)

    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped