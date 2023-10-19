from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import time
from datetime import timedelta

'''
This module is for preparing the map matched data for further analyses and impedance calibration:
    
Calculate shortest path from all o's to all d's to determine detour percent of chosen trip and overlap
Remove incomplete matches
Add user info for segmenting data

Segment data
    clustering
    by stated rider type labels

Export segmented data for optimization step

'''

fp = Path.home() / "Documents/GitHub/Impedance-Calibration"

#%%import dict of matched traces with link ids and geometry 
with (fp/'matched_traces.pkl').open('rb') as fh:
     matched_traces = pickle.load(fh)

# remove trips with less than 95% matching
matched_traces = {key:matched_traces[key] for key in matched_traces.keys() if matched_traces[key]['match_ratio'] >= 0.95}

# get match length
matched_length = {key:matched_traces[key]['matched_trip'].length.sum() for key in matched_traces.keys()}

#%% import trips to attach data to
trips_df = pd.read_csv(fp/'trips.csv')

#reduce tripsdf size
trips_df = trips_df[trips_df['tripid'].isin(set(matched_traces.keys()))]

#%% find new start/end coords

# find euclidean distance between start and end coord (for finding loop trips)
start_geo = gpd.points_from_xy(trips_df['start_lon'],trips_df['start_lat'],crs='epsg:4326').to_crs('epsg:2240')
end_geo = gpd.points_from_xy(trips_df['end_lon'],trips_df['end_lat'],crs='epsg:4326').to_crs('epsg:2240')
trips_df['euclidean_distance'] = start_geo.distance(end_geo)

#%% import trip csv

# import user info and filter columns
export_fp = Path.home() / 'Downloads/cleaned_trips'
trip = pd.read_csv(export_fp/"trip.csv", header = None)
col_names = ['tripid','userid','trip_type','description','starttime','endtime','num_points']
trip.columns = col_names
trip.drop(columns=['starttime','endtime','num_points'],inplace=True)

trips_df = pd.merge(trips_df,trip,on='tripid')

#%% import user csv
user = pd.read_csv(export_fp/"user.csv", header=None)
user_col = ['userid','created_date','device','email','age','gender','income','ethnicity','homeZIP','schoolZip','workZip','cyclingfreq','rider_history','rider_type','app_version']
user.columns = user_col
user.drop(columns=['device','app_version','app_version','email'],inplace=True)

# merge trip and users
#join the user information with trip information
trips_df = pd.merge(trips_df,user,on='userid')

#%% remove loops and exercies trips
tolerance_ft = 1000
trips_df = trips_df[trips_df['euclidean_distance']>tolerance_ft]
trips_df = trips_df[trips_df['trip_type']!='Exercise']

#%%

#retrieve the ods for shortest path routing 
ods = { key : (int(matched_traces[key]['nodes'][0]),int(matched_traces[key]['nodes'][-1])) for key, item in matched_traces.items()}

#add start and end node to trips_df
trips_df['od'] = trips_df['tripid'].map(ods)

#remove if o==d
def remove_self_loops(tup):
    if tup[0]==tup[1]:
        return False
    else:
        return True
trips_df = trips_df[trips_df['od'].apply(lambda tup: remove_self_loops(tup))]

#turn into set to remove duplicate ods
sources = set([x[0] for key, x in ods.items()])
ods = set([x for key, x in ods.items()])

#import network links
links = gpd.read_file(fp/"final_network.gpkg",layer='links')
links['A'] = pd.to_numeric(links['A'])
links['B'] = pd.to_numeric(links['B'])
links['length_ft'] = links.length
links['tup'] = list(zip(links['A'],links['B']))

#make graph
G = nx.DiGraph()  # create directed graph
for row in links[['A','B','length_ft']].itertuples(index=False):
    G.add_weighted_edges_from([(int(row[0]), int(row[1]), int(row[2]))],weight='weight')   

#%% Find shortest path to single-source with target

'''
This ends up being faster because most bike trips are going to be very short and this graph is large

dijkstra in this case terminates when the target is reached

need a custom dijkstra that returns other nodes visited to prevent redoing them but will probably need help
'''

algorithm_name = "Single Source Dijktra Multiple Times"

shortest_paths = {}
print(f'Shortest path routing with {algorithm_name}')
start_time = time.perf_counter()
for source in tqdm(sources):
    targets = [x[1] for x in ods if x[0] == source]
    #perform shortest path routing for all target nodes from source node (from one to all until target node has been visited)
    for target in targets:  
        length, node_list = nx.single_source_dijkstra(G,source,target,weight='weight')
        edge_list = [ (node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
        shortest_paths[(source,target)] = {'edge_list':edge_list,'length':length}
        
duration = timedelta(seconds=time.perf_counter()-start_time)
print(f'{algorithm_name} took {duration}')

with (fp/'shortest_paths.pkl').open('wb') as fh:
     pickle.dump(shortest_paths,fh)

#%% calculate detour rate and overlap

'''
Detour percent is length of chosen path minus length of the shortest path divided by the shortest path times 100

Overlap is the total length of links shared between the chosen and generated route divided by length of chosen route

Overlap does not currently count looping back on the same link unless it is in the reverse direction


>>> [i for i, j in zip(a, b) if i == j]
[5]

'''

#create dict of link lengths for lookup
link_lengths = dict(zip(links['tup'],links['length_ft']))

print('Calculating overlap and lengths')
for key, item in tqdm(shortest_paths.items()):
    #add shortest path length to trips_df
    trips_df.loc[trips_df['od']==key,'shortest_length'] = item['length']
    
    #get trip ids corresponding to ods
    tripids = trips_df.loc[trips_df['od']==key,'tripid'].to_list()
    
    #calculate overlap
    for tripid in tripids:
        chosen_edges = matched_traces[tripid]['edges']
        chosen_edges = [(int(link1),int(link2)) for link1,link2 in chosen_edges]
        chosen_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_edges])
        trips_df.loc[trips_df['tripid']==tripid,'chosen_length'] = chosen_length
        
        chosen_and_shortest = set(chosen_edges) & set(item['edge_list'])
        overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
        trips_df.loc[trips_df['tripid']==tripid,'overlap_length'] = overlap_length

# calculate detour percent (if negative something then wrong way travel was involved)
trips_df['detour_rate'] = ((trips_df['chosen_length'] - trips_df['shortest_length']) / trips_df['shortest_length'] * 100).round(0)

# export
trips_df.to_csv(fp/'trips_df_postmatch.csv')


'''
may need to also remove trips that are repeats by the same person especially if the route didn't change


also need to consider high detour cases, where overlap isn't a good measure??
maybe a distance + overlap matching is best?


'''



#%% Floyd Warshal - runs into a memory error, would have to simplify network

# algorithm_name = "Floyd Warshall"

# print(f'Shortest path routing with {algorithm_name}')
# start_time = time.perf_counter()
# predcessor, distance = nx.floyd_warshall_predecessor_and_distance(G, weight='weight')
# duration = timedelta(seconds=time.perf_counter()-start_time)
# print(f'{algorithm_name} took duration')
# timings[algorithm_name] = duration
    

#%% Multi-Source dijkstra - only returns the shortest path from one source, does not report all?

# algorithm_name = "Multi-Source Dijkstra"

# print(f'Shortest path routing with {algorithm_name}')
# start_time = time.perf_counter()
# distance, path = nx.multi_source_dijkstra(G,sources=list(sources),weight='weight')
# duration = timedelta(seconds=time.perf_counter()-start_time)
# print(f'{algorithm_name} took {duration}')
# timings[algorithm_name] = duration



#%%
# algorithm_name = "Single Source Dijktra Once"
# shortest_paths_edges = {}
# shortest_paths_lengths = {}

# print(f'Shortest path routing with {algorithm_name}')
# start_time = time.perf_counter()

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
#         shortest_paths_edges[(source,target)] = edge_list
#         shortest_paths_lengths[(source,target)] = length

# duration = timedelta(seconds=time.perf_counter()-start_time)
# print(f'{algorithm_name} took {duration}')
# timings[algorithm_name] = duration




# #%% single source

# # do shortest path routing with distance based impedance
# shortest_paths_edges = {}
# shortest_paths_lengths = {}
# print('Shortest path routing with single source dijkstra')
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
#         shortest_paths_edges[(source,target)] = edge_list
#         shortest_paths_lengths[(source,target)] = length

# # save dict
# with (fp/'shortest_paths_edges.pkl').open('wb') as fh:
#      pickle.dump(shortest_paths_edges,fh)
# with (fp/'shortest_paths_lengths.pkl').open('wb') as fh:
#      pickle.dump(shortest_paths_lengths,fh)
     
# #%% load dict
# with (fp/'shortest_paths.pkl').open('rb') as fh:
#      shortest_paths_edges = pickle.load(fh)
# # with (fp/'shortest_paths_lengths.pkl').open('wb') as fh:
# #      shortest_paths_lengths = pickle.load(fh)
     




