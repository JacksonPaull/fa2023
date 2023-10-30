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

def init_population(mode,N, m, networks):
    
    #TODO: initialize a population of N nodes 

    #depending on which mode, choose m patient 0's 
    if mode == 'top_degree':
        # choose 5 nodes with the highest degree from networks[0]
    else if mode == 'random':
        #choose 5 nodes at random 

    return 

###########################################
#              SIR Simulation  
###########################################
 
def run_experiment(days,networks):
    
    for day in range(days):
        G = networks[day]
        
        #susceptible to infected
        
        #for person in infected:
            #get neighbors
            neighbors = list(G.neighbors(person))
            #for each neighbor: 
                #apply chance of infection
                #if chance <= p_transmit_virus:
                    #this node is now infected 
                    #set recovery time to day + DELTA_RECOVER
            

        #infected to recovered 
        #Transition infected people to recovered after DELTA_RECOVER days.

    return 

###########################################
#              Plot Results  
###########################################

def plot_SIR():
    plt.figure()

    #plot infection vs day from part A 

    #plot infection vs day from part B

    #plot infection vs day from part C mitigation on 'degree' initialization

    #plot infections vs day from part C mitigation on 'random' initialization

    #label your plot lines 
    plt.legend()
    plt.grid()
    plt.title("Number of Infections vs Day")
    plt.xaxis("Day")
    plt.yaxis("Number of infections")
    plt.show()
    #or plt.savefig(fname) to save your figure to a file 

    #screenshot this figure and place in your HW2 PDF. 
    return 


def main():
    
    #init experiment based on these parameters:
    days = 30 
    N = 1000 
    m = 5 

    networks = build_networks(days,N)
    
    #initialize population for top 5 degrees 
    init_population("degree", N, m, networks)

    #run SIR experiment A
    run_experiment( days,networks)
   
    #initialize population for random 5 nodes 
    init_population("random", N, m, networks)

    #run SIR experiment B
    run_experiment(days,networks)
   

    #implement your own mitigation strategy (part C)
    #Note: use your mititgation strategy on top "degree" initialization
    # and"random" initialization to compare. 

    #plot all infections vs time 
    plot_SIR()




if __name__ == '__main__':
    main()
