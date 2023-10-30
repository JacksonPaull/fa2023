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

def mitigate(networks):
    # TODO Apply a mitigation strat
    pass

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

def plot_SIR(partA, partB, partC1, partC2, figname='./experiment_results.png', show=True, save=True):
    plt.figure()

    # Plots
    plt.plot(partA, label='Top M, no mitigation (Part A)')
    plt.plot(partB, label='Random, no mitigation (Part B)')
    plt.plot(partC1, label='Top M, mitigated (Part C)')
    plt.plot(partC2, label='Random, mitigated (Part C)')

    #label your plot lines 
    plt.legend()
    plt.grid()
    plt.title("Number of Infections vs Day")
    plt.xlabel("Day")
    plt.ylabel("Number of infections")

    if show:
        plt.show()
    if save:
        plt.savefig(figname)
    #or plt.savefig(fname) to save your figure to a file 

    #screenshot this figure and place in your HW2 PDF. 
    return 


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
    # Mitigate

    partC1 = run_experiment(days, networks)

    init_population('random', N, m, networks)
    # Mitigate

    partC2 = run_experiment(days, networks)

    #plot all infections vs time 
    plot_SIR(pA, pB, partC1, partC2)



if __name__ == '__main__':
    main()
