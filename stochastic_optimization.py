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
from datetime import timedelta
import ast

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

Todo:
Using initial betas to see how well they recreate routes


'''

fp = Path.home() / "Documents/GitHub/Impedance-Calibration"
with (fp/"calibration_results.pkl").open('rb') as fh:
    results = pickle.load(fh)


#%%

def objective_function(betas,links,G,ods,trips_df,matched_traces,durations):
    
    start_time = time.perf_counter()
    
    #beta coefficients
    betas = np.asarray(betas)

    #use beta coefficients to calculate link costs
    #links['cost'] = links['length_ft']*(1+betas[0]*links['high_stress']) + 99999999999999999
    
    #links['cost'] = links.apply(lambda row: cost_function(betas,row),axis=1)
    
    attr = betas[0]*links['notBeltLine']+betas[1]*links['not_infra']+betas[2]*links['twolanes']+betas[3]*links['here_>30mph']
    links['cost'] = links['length_ft']*(1+attr) + 99999999999999
    #links['cost'] = links['length_ft']*(1+betas[0]*links['high_stress']) + 99999999999999999

    #heading, rider_type

    #turn into dict
    weights = dict(zip(links['tup'],links['cost']))

    #update edge weights
    nx.set_edge_attributes(G,values=weights,name='weight')
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')
    sources = list(set([x[0] for x in ods]))
    for source in sources:
        targets = list((set([x[1] for x in ods if x[0] == source])))
        #perform shortest path routing for all target nodes from source node (from one to all until target node has been visited)
        for target in targets:  
            try:
                length, node_list = nx.single_source_dijkstra(G,source,target,weight='weight')
                edge_list = [(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
                shortest_paths[(source,target)] = {'edge_list':edge_list,'length':length}
            except:
                continue

    #calculate overlap
    #print('Calculating overlap')
    all_overlap = 0
    for idx, row in trips_df.iterrows():

        try:

            modeled_edges = shortest_paths[row['od']]['edge_list']     
    
            chosen_edges = matched_traces[row['tripid']]['edges']
            chosen_edges = [(int(link1),int(link2)) for link1,link2 in chosen_edges]
            
            chosen_and_shortest = set(chosen_edges) & set(modeled_edges)
            
            overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
            all_overlap += overlap_length
        
        except:
            continue

    #calculate objective function value
    val = -1 * all_overlap / sum_all
    print(val)
    
    duration = timedelta(seconds=time.perf_counter()-start_time)
    durations.append(duration)
    
    return val


#%%
fp = Path.home() / "Documents/GitHub/Impedance-Calibration"

#import dict of matched traces with link ids and geometry 
with (fp/'matched_traces.pkl').open('rb') as fh:
     matched_traces = pickle.load(fh)

#import segment to use
segment_filepaths = list((fp/'segments').glob('*'))

results = {}

#import network links
links = gpd.read_file(fp/"final_network.gpkg",layer='links')
links['A'] = pd.to_numeric(links['A'])
links['B'] = pd.to_numeric(links['B'])
links['length_ft'] = links.length
links['cost'] = links.length

#%%

# #start all links as low stress
# links['high_stress'] = 0

# # All primary/secondary roads stressful unless cycletrack/multiuse
# primary_secondary = ['primary','secondary','primary_link','secondary_link','trunk','trunk_link']
# links.loc[(links['highway'].isin(primary_secondary)) & (links[['osm_pbl','osm_mu']].sum(axis=1) == 0),'high_stress'] = 1

# # tertiary stressful unless bike lane +
# tertiary = ['tertiary','tertiary_link']
# links.loc[(links['highway'].isin(tertiary)) & (links[['osm_pbl','osm_mu','osm_bl']].sum(axis=1) == 0),'high_stress'] = 1

# Unclassified/residential are low stress for now

#%%

#have beltline variable
links.loc[links['name']!='Atlanta BeltLine Eastside Trail','notBeltLine']=1
links.loc[(links[['osm_mu','osm_pbl','osm_bl']]==0).any(axis=1),'not_infra'] = 1
links.loc[(links[['here_2-3lpd','here_>4lpd']]==1).any(axis=1),'twolanes'] = 1
links.fillna(0,inplace=True)

# #%%
# def cost_function(betas,link):
#     return np.matmul(np.asarray(link[['notBeltLine','not_infra','twolanes','here_>30mph']]),betas)[0]

#     #stress = betas * np.asarray(links[['not_infra','here_25-30mph','here_>30','here_2-3lpd','here_>4lpd','notBeltLine'])
    
#     # ['temp_ID', 'here_<25mph', 'here_25-30mph',
#     #    'here_>30mph', 'here_1lpd', 'here_2-3lpd', 'here_>4lpd', 'ST_NAME',
#     #    'FUNC_CLASS', 'DIR_TRAVEL']

#%%

#make graph
G = nx.DiGraph()  # create directed graph
for row in links[['A','B','cost']].itertuples(index=False):
    G.add_weighted_edges_from([(int(row[0]), int(row[1]), int(row[2]))],weight='weight')

for segment_filepath in segment_filepaths:
    trips_df = pd.read_csv(segment_filepath)
    
    trips_df['od'] = trips_df['od'].apply(lambda row: ast.literal_eval(row))
    
    #inputs
    sum_all = trips_df['chosen_length'].sum() * 5280
    links['tup'] = list(zip(links['A'],links['B']))
    link_lengths = dict(zip(links['tup'],links['length_ft']))
    durations = []
    ods = list(set(trips_df['od'].tolist()))
    
    start = time.time()
    bounds = [[-5,5],[-5,5],[-5,5],[-5,5]]
    x = minimize(objective_function, bounds, args=(links,G,ods,trips_df,matched_traces,durations), method='pso')
    end = time.time()
    print(f'Took {(end-start)/60/60} hours')
    results[segment_filepath] = (x.x,x.fun)

#%%

with (fp/"calibration_results.pkl").open('wb') as fh:
    pickle.dump(results,fh)

#%%

new_results = {key.parts[-1].split('.csv')[0]:items for key, items in results.items()}
new_results = pd.DataFrame.from_dict(new_results,orient='index',columns=['coefs','overlap'])
new_results[['not_beltline','not_infra','2lanes','30mph']] = new_results['coefs'].apply(pd.Series)

#%% test params

# betas = np.array([0.03,-3.89])

# links['cost'] = links.apply(lambda link: link_cost(link,betas),axis=1)

# #update edge weights
# nx.set_edge_attributes(G,values=weights,name='weight')

# #do shortest path routing
# shortest_paths = {}
# print(f'Shortest path routing with coefficients: {betas}')
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