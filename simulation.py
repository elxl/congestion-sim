import random, math, pickle
import argparse, logging
import numpy as np
from structures import Network
from functions import initialize_demand, initialize_vehicle, matching, optimization
from collections import defaultdict
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description="Congestion-Sim")

# Simulation parameters
parser.add_argument(
    "--congestion_level",
    type=float,
    default=0,
    help="congestion level (default: 0)"
)
parser.add_argument(
    "--fleet_size",
    type=int,
    default=2000,
    help="number of vehicles (default 2000)"
)
parser.add_argument(
    "--log_path",
    type=str,
    default="log/log",
    help="Log file path"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="output/output",
    help="Output file path"
)

args = parser.parse_args()
logging.basicConfig(filename=args.log_path+".log",
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

np.random.seed(42)

fleet_size = args.fleet_size
congestion_level = args.congestion_level
net = Network(congestion_level)
# pickle.dump(net, open("Network.pkl",'wb'))

# Initialize demand and vehicle objects
logging.info("Intializing passengers and vehicles ...")
demand_list, demand_id_dict = initialize_demand(net)
vehicle_list, vehicle_id_dict = initialize_vehicle(fleet_size, net)

simulation_start_time = datetime(2019,6,net.date,net.start_time[0],0,0)
simulation_end_time = datetime(2019,6,net.date,net.end_time[0],0,0)
simulation_time = simulation_start_time
logging.info("Intializing finish. Simulation starts.")

while True:
    if simulation_time >= simulation_end_time:
        break

    time_index = int((simulation_time - simulation_start_time).total_seconds() / net.time_interval_length)
    number_of_intervals = int(net.rebalancing_time_length / net.time_interval_length) # number of intervals in one optimization step
    end_time_index = time_index + min(net.d[:, :, time_index:].shape[2], number_of_intervals) # end time index for current optimization step

    P_matrix = net.P[:,:,time_index:end_time_index] # Probility of vehicle staying occupied
    Q_matrix = net.Q[:,:,time_index:end_time_index] # Probability of occupied vehicle becomes vacant

    # Find initial occupied & vacant vehicle distributions
    V_init = np.zeros(net.n) # vacant vehicles
    O_init = np.zeros(net.n) # occupied vehicles

    zone_vacant_veh_dict = defaultdict(list)
    for veh in vehicle_list:
        veh_loc = veh.current_location
        vehicle_zone = net.road_node_to_zone_dict[veh_loc]
        if veh.status == 3:
            O_init[vehicle_zone] += 1 # zone index from 0 to 62
        elif veh.status == 0:
            V_init[vehicle_zone] += 1
            zone_vacant_veh_dict[vehicle_zone].append(veh.id)

    logging.info(f"{simulation_time}: Rebalancing")

    K_sub = end_time_index - time_index
    a_sub = net.a[:,:,time_index:end_time_index] # if traveling time is bigger than rebalancing threshold
    b_sub = net.b[:,:,time_index:end_time_index] # if traveling time is bigger than maximum waiting time
    d_sub = net.d[:,:,time_index:end_time_index] # zone centroids distance 

    r = net.true_demand[:, time_index:end_time_index]
    rebalancing_decision = optimization(r, V_init, O_init, P_matrix, Q_matrix, net.n, K_sub, a_sub, b_sub, d_sub, net.β, net.γ)

    rebalancing_decision = (np.floor(rebalancing_decision[:,:,0])).astype(int)

    # Rebalancing vacant vehicles
    road_update = defaultdict(int) # Dictionary to keep track of the number of vehicles to be updated on roads
    for i in range(net.n):
        for j in range(net.n):
            rebalancing_veh_number = rebalancing_decision[i,j]
            if rebalancing_veh_number <= 0:
                continue
            rebalancing_veh_list = random.sample(zone_vacant_veh_dict[i], rebalancing_veh_number)
            for veh_id in rebalancing_veh_list:
                veh = vehicle_id_dict[veh_id]
                random_number = 0
                reb = True
                while True:
                    dest_node = random.choice(net.zone_to_road_node_dict[j])
                    rebalancing_dist = net.road_distance_matrix[veh.current_location, dest_node]
                    rebalancing_time = net.travel_time(veh.current_location, dest_node)
                    if rebalancing_time <= net.maximum_rebalancing_time:
                        break
                    random_number += 1
                    # Sample 10 times to get points in the two zones between which the traveling time is less than the maximum
                    if random_number >= 10:
                        reb = False
                        break
                if reb:
                    # Update information of rebalanced vehicles
                    veh.status = 1
                    veh.rebalancing_travel_distance.append(rebalancing_dist)
                    veh.rebalancing_trips.append(1)
                    veh.location = dest_node
                    veh.available_time = simulation_time + timedelta(seconds=(int(math.floor(rebalancing_time))))

    # update current location for rebalanced vehicles
    for veh in vehicle_list:
        if veh.status == 1:
            trip_path = net.get_path(veh.current_location,veh.location)
            # Deduct vehicle from the current segment
            road_update[(veh.last_location,veh.current_location)] -= 1
            travel_time = 0
            arrive = True
            for i in range(len(trip_path) - 1):
                start_node = trip_path[i]
                end_node = trip_path[i+1]
                segment_time = net.roads[(start_node,end_node)].travel_time()
                travel_time += segment_time
                veh.last_location = start_node
                if travel_time > net.time_interval_length:
                    veh.current_location = end_node 
                    # Add vehicle to the new segment
                    road_update[(start_node, end_node)] += 1
                    if i != len(trip_path) - 2:
                        arrive = False
                    break
            if arrive:
                veh.current_location = veh.location
                veh.available_time = simulation_time + timedelta(seconds=(int(math.floor(travel_time))))
                veh.status = 0

    # Update road flow
    for (o,d),num in road_update.items():
        if o!=d:
            net.roads[(o,d)].flow += num
    for road in net.roads.values():
        road.updated = False

    # Matching engine in the simulation
    matching_simulation_time = simulation_time
    while True:
        
        logging.info(f"{matching_simulation_time}:Matching")
        if matching_simulation_time >= simulation_time + timedelta(seconds=net.time_interval_length):
            break

        available_vehicles = []
        for veh in vehicle_list:
            if (veh.status == 0) and (veh.available_time < matching_simulation_time + timedelta(seconds=net.matching_window)):
                available_vehicles.append(veh)

        requesting_demands = []
        for dem in demand_list:
            if simulation_start_time <= dem.request_time < matching_simulation_time + timedelta(seconds=net.matching_window):
                if dem.assign_time is None:
                    if not dem.leave_system:
                        requesting_demands.append(dem)

        matching_list, unserved_pax_list = matching(available_vehicles, requesting_demands, net)

        # Update Passengers not matched
        for pax_id in unserved_pax_list:
            pax = demand_id_dict[pax_id]
            pax.wait_time += net.matching_window
            if pax.wait_time >= net.maximum_waiting_time:
                pax.leave_system = True

        matched_veh = []
        for ((veh_id, pax_id), pickup_dist) in matching_list:

            # Passenger choice
            pax = demand_id_dict[pax_id]
            p = net.price(pax.origin,pax.destination)
            t = net.travel_time(pax.origin,pax.destination)
            accept = pax.accept(p,t,net.base_price[pax.origin,pax.destination],net.base_time[pax.origin,pax.destination])
            if not accept:
                pax.wait_time += net.matching_window
                if pax.wait_time >= net.maximum_waiting_time:
                    pax.leave_system = True

                continue               

            # Update matched vehicle and passenger information if the trip is accepted
            veh = vehicle_id_dict[veh_id]
            matched_veh.append(veh_id)
            pax.assign_time = matching_simulation_time + timedelta(seconds=net.matching_window) # vehicle is matched with the passenger in next time step

            veh.passenger_id = pax_id
            veh.location = pax.origin
            veh.status = 2
            veh.served_passenger.append(pax.id)
            earning = net.earning(pax.origin,pax.destination)
            veh.trip_earning.append(earning)
            veh.pickup_travel_distance.append(pickup_dist)
            veh.occupied_travel_distance.append(net.road_distance_matrix[pax.origin, pax.destination])

        # update current location for matched vehicles
        road_update = defaultdict(int)
        for veh in vehicle_list:
            if (veh.status == 2) and (veh.id not in matched_veh):
                # Pick up passengers
                trip_path = net.get_path(veh.current_location,veh.location)
                road_update[(veh.last_location,veh.current_location)] -= 1
                travel_time = 0
                arrive = True
                for i in range(len(trip_path) - 1):
                    start_node = trip_path[i]
                    end_node = trip_path[i+1]
                    segment_time = net.roads[(start_node,end_node)].travel_time()
                    travel_time += segment_time
                    veh.last_location = start_node
                    if travel_time > net.matching_window:
                        veh.current_location = end_node
                        road_update[(start_node, end_node)] += 1
                        if i != len(trip_path) - 2:
                            arrive = False
                        break
                if arrive:
                    pax = demand_id_dict[veh.passenger_id]
                    veh.current_location = veh.location
                    veh.location = pax.destination
                    pax.wait_time = travel_time + datetime.timestamp(matching_simulation_time) - datetime.timestamp(pax.request_time)
                    pax.pickup_time = matching_simulation_time + timedelta(seconds=net.matching_window)
                    veh.status = 3
            elif veh.status == 3:
                # Deliver passengers
                trip_path = net.get_path(veh.current_location,veh.location)
                road_update[(veh.last_location,veh.current_location)] -= 1
                travel_time = 0
                arrive = True
                for i in range(len(trip_path) - 1):
                    start_node = trip_path[i]
                    end_node = trip_path[i+1]
                    segment_time = net.roads[(start_node,end_node)].travel_time()
                    travel_time += segment_time
                    veh.last_location = start_node
                    if travel_time > net.matching_window:
                        veh.current_location = end_node
                        road_update[(start_node, end_node)] += 1
                        if i != len(trip_path) - 2:
                            arrive = False
                        break
                if arrive:
                    veh.current_location = veh.location
                    veh.available_time = matching_simulation_time + timedelta(seconds=(int(math.floor(travel_time))))
                    veh.status = 0
                    pax = demand_id_dict[veh.passenger_id]

                    pax.travel_time = travel_time + datetime.timestamp(matching_simulation_time) - datetime.timestamp(pax.request_time) - pax.wait_time
                    pax.arrival_time = matching_simulation_time + timedelta(seconds=(int(math.floor(travel_time))))

        # Update road flow
        for (o,d),num in road_update.items():
            if o!=d:
                net.roads[(o,d)].flow += num
        for road in net.roads.values():
            road.updated = False

        matching_simulation_time += timedelta(seconds=net.matching_window)
        net.reset_price()

    simulation_time += timedelta(seconds=net.time_interval_length)

logging.info("Simulation Ends")

output = dict()
# Output simulation results
vehicle_served_passenger_list = []
vehicle_earning_list = []
vehicle_occupied_dist_list = []
vehicle_pickup_dist_list = []
vehicle_rebalancing_dist_list = []
vehicle_rebalancing_trip_list = []
for veh in vehicle_list:
    vehicle_served_passenger_list.append(veh.served_passenger)
    vehicle_earning_list.append(sum(veh.trip_earning))
    vehicle_occupied_dist_list.append(veh.occupied_travel_distance)
    vehicle_pickup_dist_list.append(veh.pickup_travel_distance)
    vehicle_rebalancing_dist_list.append(veh.rebalancing_travel_distance)
    vehicle_rebalancing_trip_list.append(veh.rebalancing_trips)

output["vehicle_served_passenger"] = vehicle_served_passenger_list
output["vehicle_occupied_dist"] = vehicle_occupied_dist_list
output["vehicle_pickup_dist"] = vehicle_pickup_dist_list
output["vehicle_rebalancing_dist"] = vehicle_rebalancing_dist_list
output["vehicle_rebalancing_trip"] = vehicle_rebalancing_trip_list

pax_wait_time_list = []
pax_travel_time_list = []
pax_leave_list = []
pax_request_time_list = []
pax_trip_price_list = []
pax_leave_number = 0
total_pax_number = 0
for pax in demand_list:
    total_pax_number += 1
    pax_request_time_list.append(pax.request_time)
    if pax.wait_time > 0 and not pax.leave_system:
        pax_wait_time_list.append(pax.wait_time)
    if pax.travel_time > 0 and not pax.leave_system:
        pax_travel_time_list.append(pax.travel_time)
        pax_trip_price_list.append(pax.p)
    if pax.leave_system:
        pax_leave_list.append(1)
        pax_leave_number += 1

output["pax_wait_time"] = pax_wait_time_list
output["pax_travel_time"] = pax_travel_time_list
output["pax_leaving"] = pax_leave_list
output["pax_leaving_rate"] = [pax_leave_number / total_pax_number]
output["pax_request_time"] = pax_request_time_list
output["pax_trip_price"] = pax_trip_price_list

output["profit"] = sum(pax_trip_price_list) - sum(vehicle_earning_list)
# output["profit"] = sum(pax_trip_price_list) - 0.8*(np.sum(vehicle_pickup_dist_list)+np.sum(vehicle_rebalancing_dist_list)+np.sum(vehicle_occupied_dist_list))

print(f"Fleet: {fleet_size} | Profit: {output['profit']} | Unserved rate: {pax_leave_number / total_pax_number}")

with open(args.output_path + f'veh_{fleet_size}.pickle', 'wb') as handle:
    pickle.dump(output, handle)