import os
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import geopy.distance
from geopy.distance import geodesic
from pyvrp import Model
from pyvrp.stop import MaxIterations, MaxRuntime
from itertools import product
from sklearn.cluster import KMeans

class BenchmarkPolicy:
    def __init__(self, num_clusters=4, num_stores=17):
        self.data = dict()
        self.length=2000
        self.num_clusters = num_clusters
        self.num_stores = num_stores
        self.transportCost = 5;
        self.fixedTransportCost = 100;
        self.demandMean = np.array([0, 20.482191780821918, 10.865753424657534, 18.75068493150685, 13.64931506849315, 
                                    16.564383561643837, 14.043835616438356, 19.55068493150685, 17.575342465753426, 
                                    8.621917808219179, 15.764383561643836, 13.495890410958904, 13.493150684931507, 
                                    15.978082191780821, 17.873972602739727, 19.15068493150685, 12.838356164383562, 
                                    11.021917808219179])
        self.demandvariance = np.array([0, 42.73388529278946, 16.858301972000586, 26.68767123287668, 27.86019870540423, 
                                        38.658618094234406, 29.904666566310404, 41.57778112298656, 24.942796929098304, 
                                        4.219298509709474, 26.642134577751037, 13.113307240704476, 10.129760650308588, 
                                        29.581935872346808, 31.86319433990653, 25.177781122986595, 21.971052235435796, 
                                        15.999518289929243])
        self.lat = np.array([52.4572735973313,
                             52.5626752866663,
                             52.5524998075759,
                             52.5485533897899,
                             52.5491337603554,
                             52.533031250357,
                             52.5326620602486,
                             52.5257945584331,
                             52.5360673338073,
                             52.513899889604,
                             52.5006751919937,
                             52.4805171338363,
                             52.4965099365104,
                             52.4921203344399,
                             52.4575252353351,
                             52.4873876246243,
                             52.4976166075393,
                             52.4861757546093
                             ])
        self.lon = np.array([13.3878670887734,
                             13.364101495398,
                             13.3610115906129,
                             13.4127270662978,
                             13.4547634845713,
                             13.387585268824,
                             13.398873880729,
                             13.4156247242011,
                             13.435584982625,
                             13.4691377305421,
                             13.4760565571936,
                             13.4389650463,
                             13.4224855541126,
                             13.4226572154896,
                             13.3920890219171,
                             13.3764057793401,
                             13.3456685769195,
                             13.3199641186209
                             ])
        self.distance_matrix = []
        self.data['vehicle_capacity'] = 100
        self.data['num_vehicles'] = 17
        self.inventories_stores = np.zeros(self.num_stores + 1)
        self.s = 20
        self.S = 60
        self.transport_cost = 5
        self.fixed_transport_cost = 100
        self.c_holding = 1 
        self.c_lost = 10
        self.current_step = 0
        self.cost    = 0
        self.backorder_count =0
        self.onhandinventor_count=0
        self.transport_units_list = []
        self.placeholder = 0



    def generate_actions(self):
        actions = list(itertools.product(range(0, self.S + 1), repeat=self.num_clusters))
        return actions
    
    def calculate_distance_matrix(self):
            distance_matrix = np.zeros(shape=(self.num_stores + 1, self.num_stores + 1))
            for i in range(self.num_stores + 1):
                for j in range(self.num_stores + 1):
                    coords_1 = (self.lat[i], self.lon[i])
                    coords_2 = (self.lat[j], self.lon[j])
                    distance_matrix[i][j] = geopy.distance.geodesic(coords_1, coords_2).km
            self.data['distance_matrix']=distance_matrix
            return   
        
    def step(self):
        with open('C:/Users/dirk7/OneDrive - TU Eindhoven/simulation_results1.txt', 'a') as f:
            while self.current_step < self.length:
                action1=[]
                rewards = []
                for i in range(len(self.inventories_stores)):
                    if self.inventories_stores[i] <= self.s:  # Inventory level below s
                            replenish_amount = max(self.S - self.inventories_stores[i], 0)
                            action1.append(replenish_amount)
                    else:
                        action1.append(0)
                action1[0] = 0
                if sum(action1) >=0:
                    reward = self.calcDirectReward(action1)
                self._take_action(action1)
                demands = np.zeros(self.num_stores+1)
                demands = self.generate_demand(self.demandMean, self.demandvariance)
                for i in range(len(demands)):
                    self.inventories_stores[i] -= demands[i]
                    reward -= max(0, self.inventories_stores[i]) * self.c_holding + -1 * min(0, self.inventories_stores[i]) * self.c_lost
                    self.backorder_count += min(0,self.inventories_stores[i])
                    self.onhandinventor_count += max(0,self.inventories_stores[i])
                    self.inventories_stores[i] = max(0, self.inventories_stores[i])
                self.cost += reward
                self.avgCost = self.cost / self.current_step

                # Write information to the file
                f.write(f"Step {self.current_step}, "
            f"On-hand Inventory: {self.onhandinventor_count}, "
            f"Average Costs: {self.cost}, "
            f"Backorder Count: {self.backorder_count}, "
            f"Fill rate: {self.placeholder}\n")
                # Update cost and current step
                self.onhandinventor_count=0
                self.placeholder = 0
                self.backorder_count=0
                self.cost=0
                action1=[]
                reward=0
                self.current_step += 1
        
        f.close()
        return self.cost
    
    def calcDirectReward(self, action): 
            self.data['demands'] = action
            m = Model() 
            int_lat = [int(number * 100) for number in self.lat]
            int_lon = [int(number * 100) for number in self.lon]
            depot = m.add_depot(x=int_lat[0], y=int_lon[0])     
            clients = [
                m.add_client(x= int_lat[idx], y=int_lon[idx], demand=int(self.data['demands'][idx]))
                for idx in range(1, len(int_lat))
            ]
            
            locations = [depot] + clients
            
            m.add_vehicle_type(num_available=self.num_stores, capacity = self.data['vehicle_capacity'], fixed_cost = self.fixedTransportCost)
            
            for idx in range(0, len(self.lat)): 
                for jdx in range(0, len(self.lat)):
                    distance = self.data['distance_matrix'][idx][jdx]   # Manhattan
                    m.add_edge(locations[idx], locations[jdx], distance=distance)     
            res = m.solve(stop = MaxRuntime(0.0001))          
            # Append the count to the provided list
            solution = res.best
            num_trucks = len([line for line in str(solution).split('\n') if line.startswith("Route #")])
            a=sum(action)/num_trucks
            self.transport_units_list.append(a)
            self.placeholder = a
            return -1 * res.cost() 
        
    def generate_demand(self, mean, variance):
        demands = np.random.normal(loc=mean, scale=np.sqrt(variance), size=len(mean))
        demands = np.maximum(demands, 0)  # Ensure demands are non-negative
        return demands.astype(int)
        
    def _take_action(self,action):
        for i, replenish_action in enumerate(action):            
            if replenish_action > 0:  # Replenishment action       
                self.inventories_stores[i] += replenish_action  
        return


    def reset(self):
        self.inventories_stores = np.zeros(self.num_stores+1)
        self.current_step = 0
        return self.inventories_stores

# Initialize the environment
env = BenchmarkPolicy()
env.distance_matrix=env.calculate_distance_matrix()

# Generate actions
actions = env.generate_actions()

# Run episodes
episodes = 1
avg_costs = []
fill_rates = []
inventory_levels = []
backorders = []

for episode in range(episodes):
    total_reward = 0
    observation = env.reset()
    rewards = env.step()
    total_reward += rewards

    avg_costs.append(total_reward)
    inventory_levels.append(sum(env.inventories_stores))
avg_transport_units_used = sum(env.transport_units_list) / len(env.transport_units_list)
avg_oh_inv = env.onhandinventor_count/env.length
avg_bo_inv = env.backorder_count/env.length
avg_reward = total_reward/env.length

print(avg_transport_units_used)
print(avg_oh_inv)
print(avg_bo_inv)
print(avg_reward)

