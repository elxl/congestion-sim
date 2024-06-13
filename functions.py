import random, math
import numpy as np
import gurobipy as gb
from datetime import datetime, timedelta
from structures import Passenger, Vehicle

from rsome import dro, ro
from rsome import grb_solver as grb


def initialize_demand(network):
    """
    Generate passengers according to real data.

    return: list of passengers, dictionary of passengers
    """
    demand_list = []
    demand_array = np.zeros((network.n,network.end_bin-network.start_bin+1)) # Demand at each time rebalancing interval
    demand_id_dict = dict()
    demand_ind = 0

    random.seed(2023)

    for i in range(network.demand_data.shape[0]):
        request_time = datetime.strptime(network.demand_data.loc[i,"pu_time"], "%Y-%m-%d %H:%M:%S")
        if (request_time >= network.start_timestamp) & (request_time < network.end_timestamp):
            pickup_zone = network.zone_index_id_dict[int(network.demand_data.loc[i, "pu_zone"])]
            dropoff_zone = network.zone_index_id_dict[int(network.demand_data.loc[i, "do_zone"])]
            timestep = int((request_time - network.start_timestamp).total_seconds()//network.time_interval_length)
            demand_array[pickup_zone, timestep] += 1
            while True:
                pickup_node = random.sample(network.zone_to_road_node_dict[pickup_zone], 1)[0]
                dropoff_node = random.sample(network.zone_to_road_node_dict[dropoff_zone], 1)[0]
                if pickup_node != dropoff_node:
                    break

            pax = Passenger(demand_ind, pickup_node, dropoff_node, request_time, None, 0.0, 0.0, None, False)
            demand_id_dict[demand_ind] = pax
            demand_list.append(pax)
            demand_ind += 1

    return demand_list, demand_array, demand_id_dict


def initialize_vehicle(fleet_size, network):
    """
    Initialize vechiles.

    return: list of vehicles, dictionary of vehicles
    """
    vehicle_list = []
    vehicle_id_dict = dict()
    vehicle_ind = 0
    init_avail_time = datetime(2019,6,network.date,network.start_time[0],0,0)
    random.seed(2023)

    zone_vehicle_number = int(math.floor(fleet_size / network.n)) # number of vehicles in each zone
    for i in network.zone_to_road_node_dict.keys():
        road_node_list = network.zone_to_road_node_dict[i]
        vehicle_loc_list = random.choices(road_node_list, k=zone_vehicle_number) # sample number of vehicles locations in zone
        for loc in vehicle_loc_list:
            veh = Vehicle(vehicle_ind, loc, init_avail_time, loc, 0, [], [], [], [], [], [])
            vehicle_id_dict[vehicle_ind] = veh
            vehicle_list.append(veh)
            vehicle_ind += 1

    return vehicle_list, vehicle_id_dict

def get_current_location(net, matching_time, veh, demand_id_dict):
    """
    demand_id_dict: id:passenger
    """
    pax = demand_id_dict[veh.served_passenger[-1]]

    if matching_time - timedelta(seconds=net.matching_window) < pax.assign_time <= matching_time:
        return veh.current_location
    else:
        vehicle_travel_time = datetime.timestamp(matching_time) - datetime.timestamp(pax.request_time) - pax.wait_time
        veh_start_loc = pax.origin
        veh_end_loc = pax.destination
        trip_path = net.get_path(veh_start_loc,veh_end_loc)

    travel_time = 0
    for i in range(len(trip_path) - 1):
        start_node = trip_path[i]
        end_node = trip_path[i+1]
        segment_time = net.roads[(start_node,end_node)].t
        travel_time += segment_time
        if travel_time >= vehicle_travel_time:
            return end_node


def matching(vehicles, demands, network):
    """
    Match available vechiles and requests.

    vehicles: list
    demands: list

    return: matched vehicle and passenger, unserved passenger
    """
    pickup_dist = {}
    # All possible matching schemes
    for veh in vehicles:
        veh_id = veh.id
        veh_loc = veh.current_location
        for dem in demands:
            dem_id = dem.id
            dem_loc = dem.origin
            pick_distance = network.road_distance_matrix[veh_loc, dem_loc]
            if network.homo:
                pickup_time = network.travel_time_homo(veh_loc, dem_loc)
            else:
                pickup_time = network.travel_time(veh_loc, dem_loc)
            if pickup_time <= network.maximum_waiting_time:
                pickup_dist[(veh_id, dem_id)] = pick_distance

    veh_dict = {veh.id: [] for veh in vehicles}
    dem_dict = {dem.id: [] for dem in demands}

    for i in pickup_dist:
        veh_dict[i[0]].append(i)
        dem_dict[i[1]].append(i)

    # Matching optimization
    m = gb.Model("matching")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 10)
    # m.setParam('MIPGap', 0.01)

    # Decision variables
    x = {}
    for i in pickup_dist:
        x[i] = m.addVar(vtype='B', name="matching_" + str(i))

    # Decision variables for unserved matching
    y = {}
    for dem in dem_dict:
        y[dem] = m.addVar(vtype='B', name="unserved_" + str(dem))

    # Constraints
    m.addConstrs(gb.quicksum(x[i] for i in veh_dict[veh]) <= 1 for veh in veh_dict)
    m.addConstrs(gb.quicksum(x[i] for i in dem_dict[dem]) + y[dem] == 1 for dem in dem_dict)

    # Objectives
    m.setObjective(network.γ * gb.quicksum(y[i] for i in dem_dict)
                    + gb.quicksum(x[i] * pickup_dist[i] for i in pickup_dist), gb.GRB.MINIMIZE)
    m.optimize()

    # Get matched vehicle and passenger as well as the pick up distance
    matching_list = []
    for i, var in x.items():
        if var.X == 1:
            matching_list.append((i, pickup_dist[i]))

    unserved_pax_list = []
    for i, var in y.items():
        if var.X == 1:
            unserved_pax_list.append(i)

    return matching_list, unserved_pax_list
        
def optimization(r, V1, O1, P, Q, n, K, a, b, d, β, γ):

    model = ro.Model()

    # Decision variables
    x = model.dvar((n, n, K))
    y = model.dvar((n, n, K))
    O = model.dvar((n, K))
    V = model.dvar((n, K))
    S = model.dvar((n, K))
    T = model.dvar((n, K))

    # Positive variable constraints
    model.st(x >= 0,y >= 0, O >= 0,S >= 0,V >=0, T>=0)

    # Set constraints to force initial position of occupied and vacant vehicles
    model.st(V[:, 0] == V1)
    model.st(O[:, 0] == O1)

    # Set constraints related to state transitions (1,2,3)
    model.st(S == V + x.sum(axis=0) - x.sum(axis=1))
    model.st(V[i,k+1] == S[i,k] - y[:,i,k].sum() + (Q[:,i,k] * O[:,k]).sum() for i in range(n) for k in range(K-1))
    model.st(O[i,k+1] == y[:,i,k].sum() + (P[:,i,k] * O[:,k]).sum() for i in range(n) for k in range(K-1))

    model.st(x.sum(axis=1) <= V)

    # Set rebalancing constraint (4)
    model.st(a.astype(int) * x == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    model.st(y.sum(axis=0) <= S)
    model.st(y.sum(axis=0) <= r)
    model.st(T == r - y.sum(axis=1)
             )
    # Set matching constraint (10)
    model.st(b.astype(int) * y == 0)

    model.min((x * d).sum()
               + 𝛽 * (y * np.transpose(d, [1,0,2])).sum()
               + γ * T.sum())
    
    model.solve(grb, display=False)

    x_value = x.get()

    return x_value