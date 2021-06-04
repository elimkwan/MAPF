from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as constant

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import re
import pandas as pd
import copy

def rawInputToArr(scene_map="./input/random-32-32-10/random-32-32-10.map"):
#     f = open("./maze-32-32-4/maze-32-32-4.map", "r")
    f = open(scene_map, "r")
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
        
        y = np.array(np.where((k == '64')| (k == '84')))
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
    
    max_i_row = img.shape[1]-1 #33 #42
    s1 = np.arange(0.5, max_i_row+1, 0.5).round(1)
    s2 = np.tile(0.5, s1.shape[0])
    first_column = np.stack((s1,s2), axis = 1)
    first_column = np.delete(first_column, (0), axis=0)
    first_column = np.delete(first_column, (-1), axis=0)

    #add boundary to obstacle list
    max_i = img.shape[0]-1 #74
    s3 = np.tile(max_i+0.5 , s1.shape[0])
    # last_column = np.stack((s1,s3), axis = 1)
    # last_column = np.delete(last_column, (0), axis=0)
    # last_column = np.delete(last_column, (-1), axis=0)
    s4 = np.tile(max_i_row+0.5 , s1.shape[0])
    last_column = np.stack((s1,s4), axis = 1)

    first_row = np.stack((s2,s1), axis = 1)
    last_row = np.stack((s3,s1), axis = 1)
    
    final_obs = np.concatenate((obstacles, first_column, last_column, first_row, last_row))
#     final_obs =[Point(x,y) for x,y in final_obs]
    
    return final_obs, img

def rawSceneToArr(scene = "./input/random-32-32-10/scen-even/random-32-32-10-even-1.scen", num_agent=1 ):
#     f = open("./maze-32-32-4/scen-random/maze-32-32-4-random-1.scen", "r")
    f = open(scene, "r")
    lines = f.readlines()
    f.close()
    
    starts = []
    ends = []
    pairs = {}
    consider_num_agent = len(lines)-2 # how many additional start end pair
    number_of_agent = consider_num_agent
    c = 0

    # Adjust if there are more maps
    map_name = 'random-32-32-10'
    if scene.split('/')[-1][2:] == 'den101d.scen':
        map_name = 'den101d'
    elif scene.split('/')[-1][2:] == 'lak109d.scen':
        map_name = 'lak109d'
    elif scene.split('/')[-1][2:] == 'lak105d.scen':
        map_name = 'lak105d'
    
    for index, l in enumerate(lines):
        x = l.split()
        match = re.match(map_name, x[1])
        if match:
            coord = np.array(x[4:8]).astype('float') + 1
            starts.append([coord[0], coord[1]])
            ends.append([coord[2], coord[3]])
            pairs[Point(coord[0], coord[1])] = Point(coord[2], coord[3])
            
            if c == number_of_agent:
                return starts, ends, pairs
            else:
                c += 1

    print("cant find enough start end pair")
    return starts, ends, pairs


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
        # else:
            # print("removed invalid lines")
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

def addStartEndNode(occupancy_grid, pairs, nodes, edges_dir, edges, num_agent):
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
                # return  SomeDist, n # No matter how close the node is to the current node, add as new node
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
            return m
        elif case == ZeroDist:
            return c_n
        return None
    
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
        mm = appendNode2(case1, connected_node1, start)
        if mm != None:
            start_nodes.append(mm)
        mm = appendNode2(case2, connected_node2, pairs[start])
        if mm != None:
            end_nodes.append(mm)
        # start_nodes.append(connected_node1)
        # end_nodes.append(connected_node2)

        if count == num_agent:
            break
    
    # for k in new_nodes:
    #     print("new nodes", k.x, k.y)
    nodes.extend(new_nodes)   
    edges_dir.extend(new_edges_dir)
    edges.extend(new_edges)
    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped

# def generalise_start_end(occupancy_grid, start_end_pair):
#     def get_position(initial):
#         mean = [initial.x , initial.y]
#         cov = [[1, 0], [0, 100]]
#         ans = None
#         while(True):
#             ans = np.random.multivariate_normal(mean, cov)
#             #check collision
#             if not occupancy_grid.isOccupied(ans):
#                 break
#         return ans

#     # loop all position in start_end_pair dictionary
#     new_dict = {}
#     for start in start_end_pair.keys():
#         s = get_position(start)
#         e = get_position(start_end_pair[start])
#         new_dict[Point(s[0], s[1])] = Point(e[0], e[1])
#     return new_dict





def getVoronoiiParam(obstacles_loc = None, occupancy_grid = None, start_end_pair = None, num_agent=1):
    # Create Voronoi Diagram
    voronoi = Voronoi(obstacles_loc)

    # Clean Voronoi Diagram
    nodes_tmp, removed_nodes = removeInsidePoint(voronoi.vertices, occupancy_grid)
    edges_tmp = removeInsideLine(len(voronoi.vertices), voronoi.ridge_vertices, removed_nodes)
    edges_tmp = removeOtherInvalidLine(nodes_tmp, edges_tmp, occupancy_grid)

    nodes, edges_dir, edges = cleanNodesEdge(nodes_tmp, edges_tmp)

    # Generalise Start-End Point
    # if mode == "random":
    #     start_end_pair = generalise_start_end(occupancy_grid, start_end_pair)
    # elif mode == "set_to":
    #     start_end_pair = specified_start_end

    nodes, edges_dir, edges, start_nodes, end_nodes, skipped = addStartEndNode(occupancy_grid, start_end_pair, nodes, edges_dir, edges, num_agent)

    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped