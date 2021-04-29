from planner.CBS_single import cbs_single
from core.DataLoader import *
from core.DataStructure import OccupancyGrid, Point
from environment.Grid import Grid
from BayesianOptimisation.experiment_setup import Experiment
import core.Constant as constant

def exp_grid(exp):
    # obstacles_loc, image = rawInputToArr()
    # resolution = 1
    # occupancy_grid = OccupancyGrid(image, [0,0], resolution)
    # start_locations_tmp, end_locations_tmp, start_end_pair = rawSceneToArr()

    # start_locations_tmp = np.array(start_locations_tmp[:(constant.NUM_OF_AGENT+len(skipped))])
    # end_locations_tmp = np.array(end_locations_tmp[:(constant.NUM_OF_AGENT+len(skipped))])

    # start_locations = np.delete(start_locations_tmp, skipped, axis = 0)
    # end_locations = np.delete(end_locations_tmp, skipped, axis = 0) 

    if not exp.initialised:
        exp.setParameters()

    resolution = 1
    grid = Grid(exp.image, [0,0], resolution)
    paths, cost = cbs_single(grid, np.array(exp.start_locations), np.array(exp.end_locations))

    ft = cost/constant.NUM_OF_AGENT

    u = grid.getOptimiserCost(
        paths=paths, 
        end_locations=exp.end_locations, 
        total_free_space=exp.occupancy_grid.getArea())

    congestion, maxmax, avgavg = grid.getCongestionLv(
        paths=paths,
    )


    return paths, ft, u, 1, congestion, maxmax, avgavg #path, avg flowtime, utilisation, congestion, maxmax, avgavg