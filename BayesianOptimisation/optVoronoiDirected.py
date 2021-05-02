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

    def getGraph(self):
        return self.G

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
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)
            
            # if cbs_out == None:
                # print("Cant find solution with SubGraph, return high cost")
                # cbs_out = cbs(directed_voronoi, start_locations, end_locations)
                # paths, cost = cbs_out
                # global_cost, ft, ut = directed_voronoi.getOptimiser     `Cost(paths)
                # return paths, global_cost, subgraph, ft, ut, directed_voronoi_sub, 0

            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths, cost, self.start_nodes,self.end_nodes)
            if paths == None:
                paths = np.array(self.start_nodes).reshape((constant.NUM_OF_AGENT, -1))
            return paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres
        else:    
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)
            # if cbs_out == None:
                # print("Cant find solution with subgraph, return high cost")
                # return 100000

            global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths, cost, self.start_nodes,self.end_nodes)
            print("cost", global_cost)
            return global_cost

def generate_data(simulator = None, iterations = 1):
    # X = []
    # y = []

    x_path = "./data/x_path.csv"
    y_path = "./data/y_path.csv"
    ft_path = "./data/ft_path.csv"
    ut_path = "./data/ut_path.csv"
    path_path = "./data/path_path.csv"

    # x_writer = open(x_path,'a+')
    # y_writer = open(y_path,'a+')
    # ft_writer = open(ft_path,'a+')
    # ut_writer = open(ut_path,'a+')
    # path_writer = open(path_path,'a+')

    for i in range(iterations):
        # print("Generating Data, Sample ", i)

        b = 0.05 
        init_probability = np.random.random_sample(989)
        init_probability[-1] = (b - 0) * np.random.random_sample() + 0
        output = simulator.run_simulator(
            init_probability, 
            return_paths=True)


        p, global_cost, _, ft, ut, _, _ = output
        p = np.array(p)
        arr = np.repeat(i, p.shape[0]).reshape((p.shape[0],1))
        paths = np.hstack((arr , p))
        if i == 0:
            with open(x_path, 'a+') as f:
                pd.DataFrame(init_probability).to_csv(f, index=None)
            with open(y_path, 'a+') as f:
                pd.DataFrame([global_cost]).to_csv(f, index=None)
            with open(ft_path, 'a+') as f:
                pd.DataFrame([ft]).to_csv(f, index=None)
            with open(ut_path, 'a+') as f:
                pd.DataFrame([ut]).to_csv(f, index=None)
            with open(path_path, 'a+') as f:
                pd.DataFrame(paths).to_csv(f, index=None)
        else:
            with open(x_path, 'a+') as f:
                pd.DataFrame(init_probability).to_csv(f, index=None, header=None)
            with open(y_path, 'a+') as f:
                pd.DataFrame([global_cost]).to_csv(f, index=None, header=None)
            with open(ft_path, 'a+') as f:
                pd.DataFrame([ft]).to_csv(f, index=None, header=None)
            with open(ut_path, 'a+') as f:
                pd.DataFrame([ut]).to_csv(f, index=None, header=None)
            with open(path_path, 'a+') as f:
                pd.DataFrame(paths).to_csv(f, index=None, header=None)
                
    return


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

    _, ft, u1 = env.getOptimiserCost(paths, cost, exp.start_nodes, exp.end_nodes)

    # ut2 = env.getCoverage(exp)
    u2 = 0
    congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, ft, u1, u2, congestion, maxmax, avgavg, graph, subgraph, subgraph_thres

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
    b = 0.05 
    init_probability = np.random.random_sample(989)
    init_probability[-1] = (b - 0) * np.random.random_sample() + 0
    init_probability = np.round(init_probability,4)

    # 989 parameters
    # generate_data(simulator = simulateObj, iterations = 100)

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

    res = minimize(simulateObj.run_simulator, init_probability, 
    method='L-BFGS-B', jac=None, bounds= Bounds(0,1), tol=None, callback=None, 
    options={'disp': None, 'maxcor': 10, 'ftol': 10, 
    'gtol': 10, 'eps': 0.015, 'maxfun': 1, 
    'maxiter': 1, 'iprint': - 1, 'maxls': 20, 
    'finite_diff_rel_step': None})

    return res


def finalRun(
    probability = None,
    G = None,
    start_nodes = None,
    end_nodes = None):

    sim = Simulator(G, start_nodes, end_nodes)
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

    if paths == None or np.any(np.array(paths)!= end_nodes):
        print("Cant find solution with SubGraph, return None")
        paths = np.array(start_nodes).reshape((constant.NUM_OF_AGENT, -1))
        
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
        print("\nCannot find solution\n")

    _, ft, u1 = env.getOptimiserCost(paths, cost, exp.start_nodes, exp.end_nodes)

    # ut2 = env.getCoverage(exp)
    u2 = 0
    congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, ft, u1, u2, congestion, maxmax, avgavg, init_graph, subgraph, subgraph_thres


    # def run_simulator(self, probability, return_paths = False):
    #     start_locations = self.start_nodes
    #     end_locations = self.end_nodes
    #     # print("Probability", probability)
    #     self.updateEdgeProbability(self.G, np.array(probability))
    #     self.subgraph_thres = probability[-1]
    #     directed_voronoi = VoronoiDirected(self.G)
    #     directed_voronoi_sub = VoronoiDirected(self.G)
    #     env = None

    #     if return_paths:
    #         # Clean G if printing ending solution
    #         subgraph = directed_voronoi_sub.formSubGraph(
    #             thres=self.subgraph_thres, 
    #             start_nodes = self.start_nodes,
    #             end_nodes = self.end_nodes)
    #         cbs_out = cbs(directed_voronoi_sub, start_locations, end_locations)
            
    #         if cbs_out == None:
    #             print("Cant find solution with SubGraph, reverting to original graph")
    #             cbs_out = cbs(directed_voronoi, start_locations, end_locations)
    #             if cbs_out == None:
    #                 return None
                    
    #             paths, cost = cbs_out
    #             global_cost, ft, ut = directed_voronoi.getOptimiserCost(paths)
    #             global_cost += 10000 # add penality to not using subgraph
    #             return paths, global_cost, subgraph, ft, ut, directed_voronoi_sub, 0, False

    #         paths, cost = cbs_out
    #         global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths)
    #         return paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres, True
    #     else:    
    #         subgraph = directed_voronoi_sub.formSubGraph(
    #             thres=self.subgraph_thres, 
    #             start_nodes = self.start_nodes,
    #             end_nodes = self.end_nodes)
    #         cbs_out = cbs(directed_voronoi_sub, start_locations, end_locations)
    #         if cbs_out == None:
    #             print("Cant find solution with subgraph, return high cost")
    #             return 100000

    #         paths, cost = cbs_out
    #         global_cost, ft, ut = directed_voronoi_sub.getOptimiserCost(paths)
    #         # print("Global cost", global_cost)
    #         return global_cost[0]





        # else:
        #     print("Path is None, Sample", it)
        #     global_cost = 10000 # arbitary high cost
        #     if i == 0:
        #         with open(x_path, 'a+') as f:
        #             pd.DataFrame(init_probability).to_csv(f, index=None)
        #         with open(y_path, 'a+') as f:
        #             pd.DataFrame(global_cost).to_csv(f, index=None)
        #     else:
        #         with open(x_path, 'a+') as f:
        #             pd.DataFrame(init_probability).to_csv(f, index=None, header=None)
        #         with open(y_path, 'a+') as f:
        #             pd.DataFrame(global_cost).to_csv(f, index=None, header=None)
        #     it += 1

    # x_writer.close()
    # y_writer.close()
    # ft_writer.close()
    # ut_writer.close()
    # path_writer.close()









