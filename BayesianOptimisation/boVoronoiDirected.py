from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from planner.CBS import cbs
import core.Constant as constant

import seaborn as sb
import matplotlib.pyplot as plt
import GPy
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from GPyOpt.methods import BayesianOptimization

class Simulator:
    def __init__(self, graph, start_nodes, end_nodes):
        self.G = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.subgraph_thres = None

    def run_simulator(self, probability, return_paths = False):
        start_locations = self.start_nodes
        end_locations = self.end_nodes
        updateEdgeProbability(self.G, list(probability[0]))
        self.subgraph_thres = probability[0][-1]
        directed_voronoi = VoronoiDirected(self.G)
        directed_voronoi_sub = VoronoiDirected(self.G)
        env = None

        if return_paths:
            # Clean G if printing ending solution
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            cbs_out = cbs(directed_voronoi_sub, start_locations, end_locations)
            
            if cbs_out == None:
                print("Cant find solution with SubGraph, reverting to original graph")
                cbs_out = cbs(directed_voronoi, start_locations, end_locations)
                paths, cost = cbs_out
                global_cost, ft, ut = directed_voronoi.getOptimiserCost(paths)
                return paths, global_cost, subgraph, ft, ut, directed_voronoi_sub, 0

            paths, cost = cbs_out
            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths)
            return paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres
        else:    
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            cbs_out = cbs(directed_voronoi_sub, start_locations, end_locations)
            if cbs_out == None:
                print("Cant find solution with subgraph, return high cost")
                return 100000

            paths, cost = cbs_out
            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths)
            return global_cost
        
def updateEdgeProbability(graph, probability):
    i = 0
    assigned = {}
    for n in graph.nodes:
        for neighbor in graph.neighbors(n):
            if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                # print("probabilty", probability[i])
                unused = probability[i]
                graph.edges[n, neighbor, 0]['probability'] = (1-probability[i])*probability[i+1]
                graph.edges[neighbor, n, 0]['probability'] = (1-probability[i])*(1-probability[i+1])
                assigned[frozenset((n, neighbor))] = 1
                i += 2
    return i

def showSolution(result_graph, paths, image, nodes, start_nodes, end_nodes, all_path = True, path_num = 0):
    edges_in_path = []
    image2 = 1-image
    path = paths[path_num]
    for ite in range(len(path)-1):
        edges_in_path.append(np.array([path[ite],path[ite+1],0]))
        p1 = result_graph.nodes[path[ite]]['position']
        p2 = result_graph.nodes[path[ite+1]]['position']
        width = result_graph.edges[path[ite],path[ite+1],0]['capacity']
        distance = result_graph.edges[path[ite],path[ite+1],0]['distance']

        for r in (p1.x, p2.x, 1):
            for c in (p1.y, p2.y, 1):
                if (Point(r,c) in np.array(nodes)[end_nodes]):
                    image2[int(r),int(c)] = 1
                else:
                    image2[int(r),int(c)] += 0.2
    
    drawn = {}
    fig, ax = plt.subplots(figsize=(12,12))
    img = np.array(1-image)
    ax = sb.heatmap(img)
    
    loop = result_graph.edges if all_path else edges_in_path
    
    for elem in loop:
        p1 = result_graph.nodes[elem[0]]['position']
        p2 = result_graph.nodes[elem[1]]['position']
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        plt.arrow(p1.y, p1.x, dy, dx, head_width = 0.2, alpha=0.5, color = 'grey')
    
    for p in start_nodes:
        plt.scatter(nodes[p].y, nodes[p].x, color = 'red', linewidths=5)
    for p in end_nodes:
        plt.scatter(nodes[p].y, nodes[p].x, color = 'lime', linewidths=5)

    plt.gca().invert_yaxis()
    return plt

def bo_voronoi_directed(exp):

    if not exp.initialised:
        exp.setParameters()

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    simulateObj = Simulator(G, exp.start_nodes, exp.end_nodes)

    np.random.RandomState(42)
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias

    domain = []
    k = 0
    assigned = {}
    for n in G.nodes:
        for neighbor in G.neighbors(n):
            if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                unused_percentage = {
                    'name':  k, 
                    'type': 'continuous', 
                    'domain': (0,1)
                }
                direction_percentage = {
                    'name': k+1 , 
                    'type': 'continuous', 
                    'domain': (0,1)
                }
                domain.append(unused_percentage)
                domain.append(direction_percentage)
                assigned[frozenset((n, neighbor))] = 1 #tbc whether this will cause errors
                k+=2

    domain.append({
                'name':  k+1, 
                'type': 'continuous', 
                'domain': (0,0.05)
                })
        
    opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
                                domain=domain, model_type='GP', \
                                initial_design_numdata = constant.BO_INITIAL_SAMPLES,\
                                kernel=kern, acquisition_type='EI')

    out_name="bo-results.txt"
    opt.run_optimization(max_iter=constant.BO_OPT_SAMPLES, report_file=out_name)
    return opt, G

def finalRun(
    probability = None,
    G = None,
    start_nodes = None,
    end_nodes = None):

    updateEdgeProbability(G, list(probability[0]))
    subgraph_thres = probability[0][-1]
    directed_voronoi = VoronoiDirected(G)
    directed_voronoi_sub = VoronoiDirected(G)
    env = None

    # Clean Graph when printing ending solution
    subgraph = directed_voronoi_sub.formSubGraph(
        thres = subgraph_thres, 
        start_nodes = start_nodes,
        end_nodes = end_nodes)

    cbs_out = cbs(directed_voronoi_sub, start_nodes, end_nodes)

    if cbs_out == None or np.any(np.array(cbs_out[0])!= end_nodes):
        print("Cant find solution with SubGraph, reverting to original graph")
        cbs_out = cbs(directed_voronoi, start_nodes, end_nodes)
        paths, cost = cbs_out
        return paths, subgraph, directed_voronoi, 0

    paths, cost = cbs_out
    return paths, subgraph, directed_voronoi_sub, 0

def bo_voronoi_directed_clean(opt, graph, exp):
    # Form SubGraph and Regenerate Solution
    x_opt = opt.x_opt
    x_opt = x_opt.astype('float').reshape(1,len(x_opt))
    # simulateObj = Simulator(graph, exp.start_nodes, exp.end_nodes)

    paths, subgraph, env, subgraph_thres = finalRun(
        probability = x_opt,
        G = graph,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)
    
    paths_np = np.array(paths)
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find solution\n")

    _, ft, u1 = env.getOptimiserCost(paths)

    # ut2 = env.getCoverage(exp)
    u2 = 0
    congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, ft, u1, u2, congestion, maxmax, avgavg, graph, subgraph, subgraph_thres
