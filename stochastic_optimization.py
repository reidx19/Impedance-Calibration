#https://keurfonluu.github.io/stochopy/api/optimize.html

from pathlib import Path
from stochopy.optimize import minimize
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import time

'''
This module is for deriving link costs from matched GPS traces
through the stochastic optimization method used in Schweizer et al. 2020

Required Files:
Network links for shortest path routing
Matched traces

Pre-Algo:
- Create network graph with lengths as original weights
- Create dict with link lengths
- set parameter search range

Algo Steps:
Using initial beta


'''

fp = Path.home() / "Documents/GitHub/Impedance-Calibration"

#import network links
links = gpd.read_file(fp/"final_network.gpkg",layer='links')
links['length_ft'] = links.length
links['cost'] = links.length
links['tup'] = list(zip(links['A'],links['B']))

cols = ['length_ft','osm_mu']
alpha = 1

#make graph
G = nx.DiGraph()  # create directed graph
for ind, row in links.iterrows():
    G.add_weighted_edges_from([(str(row['A']), str(row['B']), float(row['cost']))],weight='weight')   

#pre-formatting of variables

#create dict of link lengths for lookup
link_lengths = dict(zip(links['tup'],links['length_ft']))

#import dict of matched traces with link ids and geometry 
with (fp/'matched_traces.pkl').open('rb') as fh:
     matched_traces = pickle.load(fh)


#retrieve the ods for shortest path routing 
ods = set([(matched_traces[key]['nodes'][0],matched_traces[key]['nodes'][-1]) for key, item in matched_traces.items()])
sources = set([x[0] for x in ods])

#%%

#do shortest path routing
shortest_paths_dist = {}
print('Shortest path routing')
for source in tqdm(sources):
    #perform shortest path routing (from one to all)
    length, paths = nx.single_source_dijkstra(G,source,weight='weight')
    #retrieve target nodes
    targets = [x[1] for x in ods if x[0] == source]
    for target in targets:
        #convert from node list to edge list
        node_list = paths[target]
        edge_list = [ (node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
        #add to dict
        shortest_paths_dist[(source,target)] = edge_list

#%% save dict
with (fp/'shortest_paths.pkl').open('wb') as fh:
     pickle.dump(shortest_paths_dist,fh)


#%%




#reduce size of matched traces for testing
random = np.random.choice(np.asarray(list(matched_traces.keys())),size=50,replace=False)
matched_traces = {key:item for key,item in matched_traces.items() if key in random}

#get denominator of objective function
sum_all = np.asarray([matched_traces[key]['matched_trip'].length.sum() for key in matched_traces.keys()]).sum()


#define parameter search range for each variable
bounds = [[0,2],[-2,2]]




#link cost function
def link_cost(link,betas):
    return (betas[0] * link['length_ft']) + (betas[1] * link['osm_mu']) + 10000000

#%% case 0 test run


# betas = [1,1]

# #beta coefficients
# betas = np.asarray(betas)

# #link cost function
# def link_cost(link,betas):
#     return (betas[0] * link['length_ft']) + (betas[1] * link['osm_mu']) + 1

# #use beta coefficients to calculate link costs
# links['cost'] = links.apply(lambda link: link_cost(link,betas),axis=1)

# #turn into dict
# weights = dict(zip(links['tup'],links['cost']))

# #update edge weights
# nx.set_edge_attributes(G,values=weights,name='weight')
# #do shortest path routing
# shortest_paths = {}

# for source in tqdm(sources):
#     #perform shortest path routing (from one to all)
#     length, paths = nx.single_source_dijkstra(G,source,weight='weight')
#     #retrieve target nodes
#     targets = [x[1] for x in ods if x[0] == source]
#     for target in targets:
#         #convert from node list to edge list
#         node_list = paths[target]
#         edge_list = [ (node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
#         #add to dict
#         shortest_paths[(source,target)] = edge_list

# #calculate overlap
# all_overlap = 0

# for key, item in matched_traces.items():
#     source = item['nodes'][0]
#     target = item['nodes'][-1]    
#     edges = item['edges']
#     modeled_edges = shortest_paths[(source,target)]
#     list_of_lists = [edges,modeled_edges]
#     overlap = list(set.intersection(*[set(list) for list in list_of_lists]))
#     overlap = [link_lengths.get(id,0) for id in overlap]
#     all_overlap += np.asarray(overlap).sum()

# #calculate objective function value
# val = -1 * all_overlap / sum_all



#%%

def objective_function(betas,links,sources,G):
    #beta coefficients
    betas = np.asarray(betas)

    #use beta coefficients to calculate link costs
    links['cost'] = links.apply(lambda link: link_cost(link,betas),axis=1)

    #turn into dict
    weights = dict(zip(links['tup'],links['cost']))

    #update edge weights
    nx.set_edge_attributes(G,values=weights,name='weight')
    
    #do shortest path routing
    shortest_paths = {}
    print('Shortest path routing')
    for source in tqdm(sources):
        #perform shortest path routing (from one to all)
        length, paths = nx.single_source_dijkstra(G,source,weight='weight')
        #retrieve target nodes
        targets = [x[1] for x in ods if x[0] == source]
        for target in targets:
            #convert from node list to edge list
            node_list = paths[target]
            edge_list = [ (node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
            #add to dict
            shortest_paths[(source,target)] = edge_list

    #calculate overlap
    all_overlap = 0
    for key, item in matched_traces.items():
        source = item['nodes'][0]
        target = item['nodes'][-1]    
        edges = item['edges']
        modeled_edges = shortest_paths[(source,target)]
        list_of_lists = [edges,modeled_edges]
        overlap = list(set.intersection(*[set(list) for list in list_of_lists]))
        overlap = [link_lengths.get(id,0) for id in overlap]
        all_overlap += np.asarray(overlap).sum()

    #calculate objective function value
    val = -1 * all_overlap / sum_all
    return val

start = time.time()
x = minimize(objective_function, bounds, args=(links,sources,G), method='pso')
end = time.time()
print(f'Took {(end-start)/60/60} hours')

#%%

