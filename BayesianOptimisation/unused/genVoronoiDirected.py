from core.DataLoader import *
from core.DataStructure import OccupancyGrid, Point
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from BayesianOptimisation.sets_of_exp import Experiment_Sets
from planner.CBS import cbs
import core.Constant as constant

import seaborn as sb
import matplotlib.pyplot as plt
import math
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
    def __init__(self, graph, start_nodes, end_nodes, exp_set, nodes):
        self.G = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.subgraph_thres = None
        self.exp_set = exp_set

        self.loc_to_key = {} 
        for n in graph.nodes:
            p = graph.nodes[n]['position']
            self.loc_to_key[(round(p.x,2),round(p.y,2))] = n

        # self.loc_to_key = {Point(round(k.x,2), round(k.y,2)): v for v, k in enumerate(nodes)}

        # for i in self.loc_to_key.keys():
        # print(self.loc_to_key.keys())

    def get_cutoff(self):
        i = 0
        assigned = {}
        for e in self.exp_set.trainable_edges:
            if (e[0],e[1]) in self.loc_to_key.keys() and (e[2],e[3]) in self.loc_to_key.keys():
                n1 = self.loc_to_key[(e[0],e[1])]
                n2 = self.loc_to_key[(e[2],e[3])]
                if n1 != n2 and frozenset((n1, n2)) not in assigned.keys() and self.G.has_edge(n1, n2, 0):
                    assigned[frozenset((n1, n2))] = 1
                    i += 1

        return i

    def get_usage(self, probabilities, m, c):
        usage = []
        for prob in probabilities:
            mse = ((prob - 0.5)**2 + (1-prob-0.5)**2)/2
            rmse = math.sqrt(mse)
            # use = 1/(1+np.exp((m * (rmse-c)))) #Sigmoid
            # print("rmse", rmse)
            use = m*rmse + c
            usage.append(use)
        return usage

    def updateEdgeProbability(self, graph, probability):
        # Only optimiser edges with length above a certain threshold
        i = 0
        assigned = {}
        idx = int(len(probability)-3)
        direction_p = probability[:idx]
        used_p = self.get_usage(direction_p, probability[-3], probability[-2])

        for e in self.exp_set.trainable_edges:
            if (e[0],e[1]) in self.loc_to_key.keys() and (e[2],e[3]) in self.loc_to_key.keys():
                n1 = self.loc_to_key[(e[0],e[1])]
                n2 = self.loc_to_key[(e[2],e[3])]
                if n1 != n2 and frozenset((n1, n2)) not in assigned.keys() and self.G.has_edge(n1, n2, 0):
                    graph.edges[n1, n2, 0]['probability'] = (used_p[i])*direction_p[i]
                    graph.edges[n2, n1, 0]['probability'] = (used_p[i])*(1-direction_p[i])
                    assigned[frozenset((n1, n2))] = 1
                    i += 1
                    # print("Fat Edge Probability", graph.edges[n, neighbor, 0]['probability'], graph.edges[neighbor, n, 0]['probability'])

        return i

    def getGraph(self):
        return self.G

    def run_simulator(self, probabilities, return_paths = False):
        probability = probabilities[0]
        start_locations = self.start_nodes
        end_locations = self.end_nodes
        # print("Probability", probability)
        self.updateEdgeProbability(self.G, np.array(probability))
        self.subgraph_thres = probability[-1]
        directed_voronoi = VoronoiDirected(self.G)
        directed_voronoi_sub = VoronoiDirected(self.G)
        env = None

        if return_paths:
            # Clean G if printing ending solution
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)

            if path == None:
                print("Padding solution to None list")
                paths = [[None]] * len(self.start_nodes)
                

            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths, cost, self.start_nodes,self.end_nodes)
            if paths == None:
                paths = np.array(self.start_nodes).reshape((len(self.start_nodes), -1))
            return paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres
        else:    
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)

            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths, cost, self.start_nodes,self.end_nodes)
            # print("global_cost", global_cost)
            return global_cost
        

def gen_voronoi_directed(exp, exp_set, first_sample = True):

    # if not exp.initialised:
        # exp.setParameters()

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    simulateObj = Simulator(G, exp.start_nodes, exp.end_nodes, exp_set, exp.nodes)
    num_probabilities = simulateObj.get_cutoff()
    loc_to_key = simulateObj.loc_to_key
    print("Number of trainable probabilities", num_probabilities)

    np.random.RandomState(42)
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias

    domain = []
    k = 0
    assigned = {}

    trainable_edges = []
    
    for e in exp_set.trainable_edges:
        if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
            n1 = loc_to_key[(e[0],e[1])]
            n2 = loc_to_key[(e[2],e[3])]
            if n1 != n2 and frozenset((n1, n2)) not in assigned.keys() and G.has_edge(n1, n2, 0):
                d = G.edges[n1, n2, 0]['capacity']
                direction_percentage = {
                    'name': k , 
                    'type': 'continuous', 
                    'domain': (0,1)
                }
                domain.append(direction_percentage)
                assigned[frozenset((n1, n2))] = 1
                k += 1

                if first_sample:
                    edge_temp = (round(G.nodes[n1]['position'].x,2), round(G.nodes[n1]['position'].y,2), round(G.nodes[n2]['position'].x,2), round(G.nodes[n2]['position'].y,2))
                    trainable_edges.append(edge_temp)
        

    #m
    domain.append({
                'name':  k+1, 
                'type': 'continuous', 
                'domain': (-0.5,0.5)
                })

    #c
    domain.append({
                'name':  k+1, 
                'type': 'continuous', 
                'domain': (0.25,0.75)
                })

    #subgraph
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

    if first_sample:
        #update trainable edges

        probabilities = opt.x_opt
        exp_set.update_trainable_edges_once(trainable_edges)

    else:
        #fill up the array
        probabilities = []
        k = 0
        assigned = {}
        for e in exp_set.trainable_edges:
            if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
                n1 = loc_to_key[(e[0],e[1])]
                n2 = loc_to_key[(e[2],e[3])]
                if n1 != n2 and frozenset((n1, n2)) not in assigned.keys():
                    if simulateObj.G.has_edge(n1, n2, 0):
                        probabilities.append(opt.x_opt[k])
                        assigned[frozenset((n1, n2))] = 1
                        k += 1
                    else:
                        probabilities.append(0.5)
            else:
                probabilities.append(0.5)

        print("last three", opt.x_opt[-3:])
        probabilities.extend(opt.x_opt[-3:])


    return probabilities, opt, G, exp_set


def finalRun(
    probability = None,
    G = None,
    start_nodes = None,
    end_nodes = None):

    sim = Simulator(G, start_nodes, end_nodes)
    cutoff_thres, num_probabilities = sim.get_cutoff()
    print("Number of trainable probabilities", num_probabilities)
    print("Length cutoff threshold", cutoff_thres)

    sim.updateEdgeProbability(G, probability)
    G = sim.getGraph()

    subgraph_thres = probability[-1]
    directed_voronoi = VoronoiDirected(G)
    directed_voronoi_sub = VoronoiDirected(G)
    env = None

    # Clean Graph when printing ending solution
    subgraph = directed_voronoi_sub.formSubGraph(
        thres = subgraph_thres, 
        start_nodes = start_nodes,
        end_nodes = end_nodes)

    paths, cost = cbs(directed_voronoi_sub, start_nodes, end_nodes)

    # if paths == None or np.any(np.array(paths)!= end_nodes):
    if paths == None:
        print("Cant find complete solution with SubGraph")
        paths = np.array(start_nodes).reshape((len(start_nodes), -1))
        
    return paths, cost, subgraph, directed_voronoi_sub, subgraph_thres



def get_results(probabilities, start, end, edge_loc, exp):
    # sim = Simulator(init_graph, start_nodes, end_nodes)
    # sim.updateEdgeProbability(init_graph, opt_probabilities)
    # final_graph = sim.getGraph()
    # directed_voronoi_sub = VoronoiDirected(final_graph)

    


    if not exp.initialised:
        exp.setParameters()

    init_graph = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    paths, cost, subgraph, env, subgraph_thres = finalRun(
        probability = opt_probabilities,
        G = init_graph,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)
    
    paths_np = np.array(paths)
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find complete solution. Some didnt reach goal\n") 

    global_cost, ft, u1 = env.getOptimiserCost(paths, cost, exp.start_nodes, exp.end_nodes)

    # ut2 = env.getCoverage(exp)
    u2 = 0
    congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, global_cost, ft, u1, u2, congestion, maxmax, avgavg, init_graph, subgraph, subgraph_thres


