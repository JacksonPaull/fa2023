#########################################
#         Problem 1 Skeleton Code
#########################################
 
#     A) plot the SIR vs time for the seed being highest degree
#     B) plot SIR vs time for  random
#     C) propose and implement your own mitigation technique
#     Note: you can change the function arguments, however 
#     keep the Simulation Parameters the same for parts A and B.

#########################################
import networkx as nx 
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import operator
import time
import random
from copy import deepcopy

#########################################
#         Simulation Parameters
#########################################

DELTA_RECOVER = 7
DELTA_INFECT = 5
p_transmit_virus = .015
SEED = .04

#########################################
#         Utility Functions
#########################################

#converts a list of tuples into dictionary 
#(use this for G.degrees() if useful)
def convert_tuple_to_dict(list_of_tuples):
    my_dict = dict()
    for i,j in list_of_tuples:
        my_dict[i] = j
    return my_dict


#build scale free networks
def build_networks(days, N):
    networks = list()
    for i in range(days):
        seed = int(SEED * N)
        G = nx.barabasi_albert_graph(N,seed)
        networks.append(G)
    return networks

###########################################
#      Initialize the Population 
###########################################

def init_population(mode, N, m, networks):
    for n in networks:
        nx.set_node_attributes(n, 'S', 'status')
        nx.set_node_attributes(n, -1, 'recovery_date')


    #depending on which mode, choose m patient 0's 
    n = networks[0]
    degrees = np.array(n.degree)
    if mode == 'degree':
        # choose 5 nodes with the highest degree from networks[0]
        order = degrees[:,1].argsort()[::-1]
        choices = degrees[order][:m,0]
        
    elif mode == 'random':
        #choose 5 nodes at random 
        order = np.random.permutation(np.arange(n.number_of_nodes()))
        choices = degrees[order][:m,0]

    else:
        raise ValueError(f'Unsupported mode \'{mode}\' specified')
        
    # Set status for initial population
    d = {n:'I' for n in choices}
    nx.set_node_attributes(n, d, 'status')

    # Set Recovery date for initial population
    d = {n:DELTA_RECOVER for n in choices}
    nx.set_node_attributes(n, d, 'recovery_date')
    return networks

def mitigate(networks, e=0.25, nn=10):
    # Apply a probability of being an 'essential person' ~25% of the population
    G = networks[0]

    essential = dict(zip(np.arange(1000), np.random.random(1000) < e))
    nx.set_node_attributes(G, essential, 'essential')

    # Non-essential people limit their interactions to 10 randomly chosen people. 
    # In practice, some of these chosen friends will also cut node out of their network
    # Some sects of the population will even disconnect 
    non_essential = [x for x, d in G.nodes(data=True) if not d['essential']]

    for node in non_essential:
        neighbors = np.array([n for n in G[node]])
        remove_neighbors = neighbors[np.argsort(np.random.random(len(neighbors)))][nn:]
        G.remove_edges_from([(node, n) for n in remove_neighbors])

    return networks

###########################################
#              SIR Simulation  
###########################################
 
def run_experiment(days,networks):
    networks = deepcopy(networks)
    for day in range(days):
        G = networks[day]

        #susceptible to infected
        infected = [x for x, d in G.nodes(data=True) if d['status'] == 'I']
        susceptible = [x for x, d in G.nodes(data=True) if d['status'] == 'S']
        
        #for person in infected:
        for i in infected:
            #get neighbors
            neighbors = list(G.neighbors(i))

            # Apply chance for infection
            d = {'status': {}, 'recovery':{}}
            for n in neighbors:
                # only consider susceptible neighbors
                if n not in susceptible:
                    continue

                if np.random.random() <= p_transmit_virus:
                    d['status'][n] = 'I'
                    d['recovery'][n] = day + DELTA_RECOVER

            nx.set_node_attributes(G, d['status'], 'status')
            nx.set_node_attributes(G, d['recovery'], 'recovery_date')


        #infected to recovered 
        #Transition infected people to recovered after DELTA_RECOVER days.
        to_recover = {x:'R' for x, d in G.nodes(data=True) if d['recovery_date'] == day}
        nx.set_node_attributes(G, to_recover, 'status')

        # Tomorrow starts where today ends (except for the last day in the simulation... for which there is no tomorrow)
        if day < days-1:
            networks[day + 1] = G.copy()

    return networks

###########################################
#              Plot Results  
###########################################

def networks_to_num_infections(networks):
    infections = []
    for N in networks:
        infected = [n for n, d in N.nodes(data=True) if d['status']=='I']
        infections.append(len(infected))

    return infections

def plot_SIR(partA, partB, partC1, partC2, figname='./experiment_results', show=True, save=True):
    fig, ax = plt.subplots()

    # Plots
    ax.plot(partA, label='Top M, no mitigation (Part A)')
    ax.plot(partB, label='Random, no mitigation (Part B)')
    ax.plot(partC1, label='Top M, mitigated (Part C)')
    ax.plot(partC2, label='Random, mitigated (Part C)')

    #label your plot lines 
    ax.legend()
    ax.grid()
    ax.set_title("Number of Infections vs Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of infections")

    if show:
        plt.show()
    if save:
        fig.savefig(figname)


def main():
    
    #init experiment based on these parameters:
    days = 30 
    N = 1000 
    m = 5 

    networks = build_networks(days,N)
    
    #initialize population for top 5 degrees, running part A
    init_population("degree", N, m, networks)
    partA = run_experiment( days,networks)
    pA = networks_to_num_infections(partA)
   
    #initialize population for random 5 nodes, running part B
    init_population("random", N, m, networks)
    partB = run_experiment(days,networks)
    pB = networks_to_num_infections(partB)

    #implement your own mitigation strategy (part C)
    #Note: use your mititgation strategy on top "degree" initialization
    # and"random" initialization to compare. 
    init_population('degree', N, m, networks)
    mitigate(networks) # Mitigate
    partC1 = run_experiment(days, networks)
    c1 = networks_to_num_infections(partC1)

    init_population('random', N, m, networks)
    mitigate(networks) # Mitigate
    partC2 = run_experiment(days, networks)
    c2 = networks_to_num_infections(partC2)

    #plot all infections vs time 
    plot_SIR(pA, pB, c1, c2)



if __name__ == '__main__':
    main()
