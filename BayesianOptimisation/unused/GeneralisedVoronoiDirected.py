from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
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
    def __init__(self, graph, start_nodes, end_nodes):
        self.G = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.subgraph_thres = None

    def get_cutoff(self):
        # Only optimiser edges with length above a certain threshold
        graph = self.G
        distance = []
        assigned = {}
        for n in graph.nodes:
            for neighbor in graph.neighbors(n):
                if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                    d = graph.edges[n, neighbor, 0]['capacity']
                    distance.append(d)
                    assigned[frozenset((n, neighbor))] = 1

        samples_to_be_considered = round(len(distance) * (constant.CONSTRAIN_PROBLEM/100))
        distance = sorted(distance, reverse=True)
        print("total number of distance", len(distance))
        # cutoff0 = distance[samples_to_be_considered]
        cutoff0 = 2

        # reversing the list
        dist = list(reversed(distance))
        # finding the index of element
        index = dist.index(cutoff0)
        # printing the final index
        final_index = len(dist) - index - 1

        self.cutoff = cutoff0
        return cutoff0, final_index

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
        for n in graph.nodes:
            for neighbor in graph.neighbors(n):
                if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                    d = graph.edges[n, neighbor, 0]['capacity']
                    if d >= self.cutoff and i < len(direction_p):
                        graph.edges[n, neighbor, 0]['probability'] = (used_p[i])*direction_p[i]
                        graph.edges[neighbor, n, 0]['probability'] = (used_p[i])*(1-direction_p[i])
                        assigned[frozenset((n, neighbor))] = 1
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
            print("global_cost", global_cost)
            return global_cost
        

def gen_voronoi_directed(exp):

    # Genereate n sets of viable start and end goals + the ultimate start/end goal we have
    # Only optimised edges that persist through the maps + the additional capacity constraints that we had



    # exp.setParameters(random = False)

    aggr_edges = []

    for num in range (10):
        exp.setParameters(mode = "random")
        new_edges = []
        for edge in exp.edges_dir:
            # print(exp.nodes[edge.prev]) #PointObject
            a = exp.nodes[edge.prev].x
            b = exp.nodes[edge.prev].y
            c = exp.nodes[edge.next].x
            d = exp.nodes[edge.next].y
            new_edges.append(frozenset({a,b,c,d}))

        print("new_edge", len(new_edges))
        if len(aggr_edges) == 0:
            aggr_edges = new_edges
        else:
            aggr_edges = list(set(aggr_edges).intersection(set(new_edges)))
        print("aggre", len(aggr_edges))


    # Find edges in graph given the aggregated set


    # G = getVoronoiDirectedGraph(
    #     occupancy_grid = exp.occupancy_grid,
    #     nodes = exp.nodes, 
    #     edges = exp.edges_dir,
    #     start_nodes = exp.start_nodes,
    #     end_nodes = exp.end_nodes)

    # Store Edges for comparison later
    # edge_list = list(G.edges(data=True))
    # print(edge_list)

    return

    # simulateObj = Simulator(G, exp.start_nodes, exp.end_nodes)
    # cutoff_thres, num_probabilities = simulateObj.get_cutoff()
    # print("Number of trainable probabilities", num_probabilities)
    # print("Length cutoff threshold", cutoff_thres)

    # np.random.RandomState(42)
    # kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    # kern_bias = GPy.kern.Bias(input_dim=2)
    # kern = kern_eq + kern_bias

    # domain = []
    # k = 0
    # assigned = {}
    # for n in G.nodes:
    #     for neighbor in G.neighbors(n):
    #         if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
    #             d = G.edges[n, neighbor, 0]['capacity']
    #             if d >= cutoff_thres and k < num_probabilities:
    #                 direction_percentage = {
    #                     'name': k , 
    #                     'type': 'continuous', 
    #                     'domain': (0,1)
    #                 }
    #                 domain.append(direction_percentage)
    #                 assigned[frozenset((n, neighbor))] = 1 #tbc whether this will cause errors
    #                 k+=1

    # #m
    # domain.append({
    #             'name':  k+1, 
    #             'type': 'continuous', 
    #             'domain': (-0.5,0.5)
    #             })

    # #c
    # domain.append({
    #             'name':  k+1, 
    #             'type': 'continuous', 
    #             'domain': (0.25,0.75)
    #             })

    # #subgraph
    # domain.append({
    #             'name':  k+1, 
    #             'type': 'continuous', 
    #             'domain': (0,0.05)
    #             })
        
    # opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
    #                             domain=domain, model_type='GP', \
    #                             initial_design_numdata = constant.BO_INITIAL_SAMPLES,\
    #                             kernel=kern, acquisition_type='EI')

    # out_name="bo-results.txt"
    # opt.run_optimization(max_iter=constant.BO_OPT_SAMPLES, report_file=out_name)
    # return opt, G


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



def get_results(opt_probabilities, exp):
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
