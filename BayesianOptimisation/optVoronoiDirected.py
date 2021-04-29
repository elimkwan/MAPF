from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from planner.CBS import cbs
import core.Constant as constant

import seaborn as sb
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# import GPy
# from emukit.core import ContinuousParameter, ParameterSpace
# from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity
# from emukit.core.initial_designs import RandomDesign
# from GPy.models import GPRegression
# from emukit.model_wrappers import GPyModelWrapper
# from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
# from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
# from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
# from GPyOpt.methods import BayesianOptimization

class Simulator:
    def __init__(self, graph, start_nodes, end_nodes):
        self.G = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.subgraph_thres = None

    def updateEdgeProbability(self, graph, probability):
        i = 0
        assigned = {}
        idx = int((len(probability)-1)/2)
        unused_p = probability[:idx]
        direction_p = probability[idx:-1]
        for n in graph.nodes:
            for neighbor in graph.neighbors(n):
                if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                    unused = unused_p[i]
                    graph.edges[n, neighbor, 0]['probability'] = (1-unused_p[i])*direction_p[i]
                    graph.edges[neighbor, n, 0]['probability'] = (1-unused_p[i])*(1-direction_p[i])
                    assigned[frozenset((n, neighbor))] = 1
                    i += 1
        return i

    def run_simulator(self, probability, return_paths = False):
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
            print("Global cost", global_cost)
            return global_cost[0]

def generate_data(simulator = None, iterations = 1):
    X = []
    y = []

    x_path = "./data/x_path.csv"
    y_path = "./data/y_path.csv"
    ft_path = "./data/ft_path.csv"
    ut_path = "./data/ut_path.csv"
    path_path = "./data/path_path.csv"

    x_writer = open(x_path,'a+')
    y_writer = open(y_path,'a+')


    for i in range(iterations):
        print("Generating Data, Sample ", i)
        b = 0.1
        init_probability = np.random.random_sample(989)
        init_probability[-1] = (b - 0) * np.random.random_sample() + 0
        paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres = simulator.run_simulator(init_probability, return_paths=True)

        X.append(init_probability)
        y.append(global_cost)

        pd.DataFrame(X).to_csv(x_writer, index=None)
        pd.DataFrame(y).to_csv(y_writer, index=None)

    x_writer.close()
    y_writer.close()
    return np.array(X),np.array(y)

        
def opt_voronoi_directed(exp):

    if not exp.initialised:
        exp.setParameters()

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    simulateObj = Simulator(G, exp.start_nodes, exp.end_nodes)

    # 989 parameters
    X, y = generate_data(simulator = simulateObj, iterations = 2)
    print(X.shape)
    print(y.shape)

    # clf = make_pipeline(StandardScaler(),
    #                 SGDRegressor(max_iter=1000, tol=1e-3))
    # clf.fit(X, y)
    # print([coef.shape for coef in clf.coefs_])


    # init_probability = []
    # bnds = []
    # k = 0
    # assigned = {}
    # for n in G.nodes:
    #     for neighbor in G.neighbors(n):
    #         if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
    #             unused_percentage = 0.1
    #             direction_percentage = 0.5
    #             init_probability.append(unused_percentage)
    #             init_probability.append(direction_percentage)
    #             bnds.append((0, 1))
    #             bnds.append((0, 1))
    #             assigned[frozenset((n, neighbor))] = 1
    #             k+=2
    # init_probability.append(0.01)
    # bnds.append((0, 0.1))
    # bnds = tuple(bnds)

    # print("Initial proabilities", init_probability)
    # minimize(simulateObj.run_simulator, init_probability, method='Nelder-Mead', tol=1e-6)
    
    # def f(x):
    #     print("x",x)
    #     if x > 5:
    #         return -3
    #     elif x < -2:
    #         return 5
    #     elif x > 20 and x < 25:
    #         return 100
    #     else:
    #         return -1000

    # res = minimize(f, 50, 
    # method='L-BFGS-B', jac=None, bounds= None, tol=None, callback=None, 
    # options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 
    # 'gtol': 1e-05, 'eps': 0.1, 'maxfun': 1000, 
    # 'maxiter': 10, 'iprint': - 1, 'maxls': 20, 
    # 'finite_diff_rel_step': None})

    return X,y#graph
