# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:01:02 2023

@author: 11481
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

class Passenger:
    def __init__(self, nid, origin, destination, request_time, assign_time, wait_time, travel_time, arrival_time, leave_system):
        self.id = nid
        self.origin = origin
        self.destination = destination
        self.request_time = request_time
        self.assign_time = assign_time
        self.wait_time = wait_time
        self.travel_time = travel_time
        self.arrival_time = arrival_time
        self.leave_system = leave_system
        self.pickup_time = None

    def accept(self,p,t,p_ref,t_ref):
        # prob = max(0,1 - (p/p_ref + t/t_ref - 2)/2)
        x = p/p_ref + t/t_ref - 2
        prob = min(1,1/(1+x*np.exp(x)))
        if np.random.uniform(0,1) <= prob:
            self.p = p
            return 1
        else:
            return 0

class Vehicle:
    def __init__(self, nid, location, available_time, current_location, status, served_passenger, trip_earning, rebalancing_travel_distance,
                 pickup_travel_distance, occupied_travel_distance, rebalancing_trips):
        """status: 0(available), 1(being rebalanced), 2(way to pick up passengers), 3(occupied) """
        self.id = nid
        self.location = location
        self.available_time = available_time
        self.current_location = current_location
        self.last_location = current_location
        self.status = status
        self.served_passenger = served_passenger
        self.trip_earning = trip_earning
        self.rebalancing_travel_distance = rebalancing_travel_distance
        self.pickup_travel_distance = pickup_travel_distance
        self.occupied_travel_distance = occupied_travel_distance
        self.rebalancing_trips = rebalancing_trips
        self.passenger_id = None

class Road:
    """ Road segment between two intersections"""

    def __init__(self, start, end, cap, t, background=0):

        self.start = start
        self.end = end
        self.cap = cap
        self.t0 = t
        self.flow = background
        self.updated = False      

    def travel_time(self):
        """ BPR """

        if not self.updated:
            self.t = self.t0 * (1 + 0.15*((self.flow)/self.cap)**4)
            self.updated = True

        return self.t
    
class Network:
    """ A complete network of New York under different congestion level"""

    def __init__(self, congestion_level=0, date=27, time_start=7, time_end=9, time_interval_length=300, rebalancing_time_length=1800, matching_window=30, capacity=1800, homo=False, maximum_wait=300):

        self.homo = homo
        self.congestion_level = congestion_level
        self.capacity = capacity
        self.free_speed = 25

        self.zone_index_id = pd.read_csv("data/NYC/zone_index_id.csv", header=None).values
        self.zone_index_id_dict = dict()
        for i in range(self.zone_index_id.shape[0]):
            self.zone_index_id_dict[self.zone_index_id[i,1]] = self.zone_index_id[i,0]
        
        self.date=date
        self.start_time = (time_start, 0) # Hour, Minute
        self.end_time = (time_end, 0) # Hour, Minute
        
        self.time_interval_length = time_interval_length # Seconds
        self.rebalancing_time_length = rebalancing_time_length # 6 time intervals
        self.matching_window = matching_window # Seconds
        data1 = pd.read_csv('data/NYC/road_network/distance.csv', header=None) / 1609.34
        self.road_distance_matrix = data1.values
        self.base_time = 3600 * self.road_distance_matrix / self.free_speed
        # self.base_price = 2.55 + 0.35*self.base_time/60 + 1.75*self.road_distance_matrix
        # self.base_price[self.base_price < 7] = 7
        self.base_price = 0.75*self.base_time/60 + 1.75*self.road_distance_matrix
        data2 = pd.read_csv('data/NYC/road_network/predecessor.csv', header=None)
        self.predecessor = data2.values
        
        self.start_bin = self.start_time[0] * 12
        self.end_bin = self.end_time[0] * 12 - 1
        self.start_timestamp = datetime(2019,6,date,self.start_time[0],0,0)
        self.end_timestamp = datetime(2019,6,date,self.end_time[0],0,0)
        
        # Demand
        data = pd.read_csv("data/NYC/processed_data/normalized_data.csv")
        data = data[(data['bin'] >= self.start_bin) & (data['bin'] <= self.end_bin)]
        prev_data = data[(data['month'] !=6) | (data['day'] < date)]
        June_date_data = data[(data['month'] ==6) & (data['day'] == date)]
        
        gd = prev_data.groupby(['day','month'])
        self.n = len(data['zone'].unique())
        K = len(data['bin'].unique())
        m = len(gd)
        data_points = np.zeros((self.n, K, m))
        index = 0
        for _, df in gd:
            y_i = df.loc[:,['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
            data_points[:,:,index] = y_i
            index += 1   

        self.data_points = data_points
            
        self.road_node_to_zone = pd.read_csv('data/NYC/road_node_to_zone.csv', header=None).values
        self.road_nodes = list(self.road_node_to_zone[:,0])
        self.zone_to_road_node_dict = defaultdict(list)
        self.road_node_to_zone_dict = dict()
        for i in range(self.road_node_to_zone.shape[0]):
            road_node_id = self.road_node_to_zone[i, 0]
            zone_id = self.road_node_to_zone[i, 1]
            self.road_node_to_zone_dict[road_node_id] = zone_id
            self.zone_to_road_node_dict[zone_id].append(road_node_id)
            
        self.zone_centriod_node = pd.read_csv("data/NYC/centroid_ind_node.csv", header=None).values
        self.centroid_to_node_dict = dict()
        for i in range(self.zone_centriod_node.shape[0]):
            self.centroid_to_node_dict[self.zone_centriod_node[i,0]] = self.zone_centriod_node[i,1]

        # Demand information used for solving optimization problems
        self.true_demand = June_date_data.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
            
        # Demand information used for simulation
        self.demand_data = pd.read_csv("data/NYC/demand/fhv_records_06272019.csv")
        
        # Problem Parameters
        self.β = 1
        self.γ = 1e2
        self.average_speed = 20
        self.maximum_waiting_time = maximum_wait    # seconds
        self.maximum_rebalancing_time = self.time_interval_length

        d = np.load("data/NYC/distance_matrix.npy") # Zone centroid distances in miles
        self.d = np.repeat(d[:, :, np.newaxis], K, axis=2) # Repeat d to create a n x n x K matrix
        # Hourly travel time
        w_hourly = np.load("data/NYC/hourly_tt.npy")
        # Hourly travel time to 288 time intervals
        a = np.repeat(w_hourly[:,:,0][:, :, np.newaxis], 12, axis=2)
        for i in range(1,24):
            b = np.repeat(w_hourly[:,:,i][:, :, np.newaxis], 12, axis=2)
            a = np.concatenate((a, b), axis=2)
        
        w = a * 3600
        w = w[:,:,self.start_bin:self.end_bin+1] # travel time matrix
        
        self.a = (w > self.maximum_rebalancing_time)
        self.b = (w > self.maximum_waiting_time)
        
        P = np.load("data/NYC/p_matrix_occupied.npy")
        Q = np.load("data/NYC/q_matrix_occupied.npy")
        self.P = np.repeat(P[:,:,np.newaxis], K, axis=2)
        self.Q = np.repeat(Q[:,:,np.newaxis], K, axis=2)

        self._generate_road_segment()
            
    def _generate_road_segment(self):

        self.roads = defaultdict(lambda: None)

        for i in self.road_nodes:
            for j in self.road_nodes:
                pre = self.predecessor[i,j]

                if not np.isnan(pre):
                    pre = int(pre)
                    self.roads[(pre,j)] = Road(pre, j, self.capacity, self.base_time[pre,j], self.congestion_level)

        # Path and price between OD. Generated on the fly to save time.
        self.paths = defaultdict(lambda: None)
        self.prices = defaultdict(lambda: None)
        self.earnings = defaultdict(lambda: None)

    def get_path(self, ori, des):

        if ori == des:
            return []

        if self.paths[(ori,des)] is not None:
            return self.paths[(ori,des)]
        else:
            trip_path = []
            temp_node = des
            # Path of vehicle
            while True:
                pred = int(self.predecessor[ori,temp_node])
                trip_path.insert(0, pred)
                if pred == ori:
                    break

                temp_node = pred
            trip_path.append(des)
            self.paths[(ori,des)] = trip_path
            return trip_path

    def travel_time(self, ori, des):
        """ Calculate travel time on edge (ori,des)"""

        if ori == des:
            return 0

        if self.paths[(ori,des)] is not None:
            trip_path = self.paths[ori,des]
        else:
            trip_path = []
            temp_node = des
            # Path of vehicle
            while True:
                pred = int(self.predecessor[ori,temp_node])
                trip_path.insert(0, pred)
                if pred == ori:
                    break

                temp_node = pred
            trip_path.append(des)
            self.paths[(ori,des)] = trip_path

        travel_time = 0
        for i in range(len(trip_path) - 1):
            start_node = trip_path[i]
            end_node = trip_path[i+1]
            segment_time = self.roads[(start_node,end_node)].travel_time()
            travel_time += segment_time

        return travel_time   
    
    def travel_time_homo(self, ori, des):

        t = self.base_time[ori,des] * (1 + 0.15*((self.congestion_level)/self.capacity)**4)
        return t


    def price(self, ori, des):
        """ Calculate estimated price on edge (ori,des)"""

        if ori == des:
            return 0
        
        if self.prices[(ori,des)] is None:
            if self.homo:
                t = self.travel_time_homo(ori, des)
            else:
                t = self.travel_time(ori, des)
            d = self.road_distance_matrix[ori, des]
            p = 0.75*t/60 + 1.75*d
            self.prices[(ori, des)] = p

        return self.prices[(ori,des)]
    
    def earning(self, ori, des):
        """ Calculate estimated driver earning on edge (ori,des)"""

        if ori == des:
            return 0
        
        if self.earnings[(ori,des)] is None:
            t = self.travel_time(ori, des)
            d = self.road_distance_matrix[ori, des]
            p = 0.287*t/(0.58*60) + 0.631*d/0.58
            self.earnings[(ori, des)] = p

        return self.earnings[(ori,des)]        

    def reset_price(self):
        self.prices = defaultdict(lambda: None)