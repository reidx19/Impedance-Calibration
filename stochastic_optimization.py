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

def create_pseudo_dual_graph(edges):
    
    #turn into graph network
    G = make_graph(edges)

    #use networkx line graph function to create pseudo dual graph
    G_line = nx.line_graph(G)
    df_line = nx.to_pandas_edgelist(G_line)

    #get rid of tuples for exporting
    df_line[['source_A','source_B']] = pd.DataFrame(df_line['source'].tolist(), index=df_line.index)
    df_line[['target_A','target_B']] = pd.DataFrame(df_line['target'].tolist(), index=df_line.index)

    #merge attributes from df
    join = edges.copy()
    join.columns = join.columns + '_x'
    df_line = df_line.merge(join,left_on=['source_A','source_B'],right_on=['source_x','target_x'])
    df_line.drop(columns=['source_x','target_x'],inplace=True)
    
    join = edges.copy()
    join.columns = join.columns + '_y'
    df_line = df_line.merge(join,left_on=['target_A','target_B'],right_on=['source_y','target_y'])
    df_line.drop(columns=['source_y','target_y'],inplace=True)
    
    #remove u-turns
    u_turn = (df_line['source_A'] == df_line['target_B']) & (df_line['source_B'] == df_line['target_A'])
    df_line = df_line[-u_turn]
    
    #change in bearing
    df_line['bearing_change'] = df_line['bearing_y'] - df_line['bearing_x']
    
    #connect lines
    df_line['geometry'] = df_line.apply(lambda row: LineString([row['geometry_x'],row['geometry_y']]),axis=1)
    df_line.drop(columns=['geometry_x','geometry_y'],inplace=True)
    
    #make all values positive
    df_line.loc[df_line['bearing_change'] < 0, 'bearing_change'] = df_line['bearing_change'] + 360 
    
    #angle here
    '''
    straight < 30 or > 330
    right >= 30 and <= 150
    backwards > 150 and less than 210
    left >= 210 and <= 270 
    
    '''
    straight = (df_line['bearing_change'] > 330) | (df_line['bearing_change'] < 30) 
    right = (df_line['bearing_change'] >= 30) & (df_line['bearing_change'] <= 150)
    backwards = (df_line['bearing_change'] > 150) & (df_line['bearing_change'] < 210)
    left = (df_line['bearing_change'] >= 210) & (df_line['bearing_change'] <= 330)
    
    df_line.loc[straight,'turn_type'] = 'straight'
    df_line.loc[right,'turn_type'] = 'right'
    df_line.loc[backwards,'turn_type'] = 'backwards'
    df_line.loc[left,'turn_type'] = 'left'
    
    #throw out backwards for now
    df_line = df_line[df_line['turn_type']!='backwards']
    
    #turn to gdf
    df_line = gpd.GeoDataFrame(df_line,crs=edges.crs,geometry='geometry')

    #psuedo graph too
    for row in df_line[['source','target']].itertuples(index=False):
        pseudo_G.add_weighted_edges_from([(row[0],row[1],0)],weight='weight') 
    
    return df_line, psuedo_G

def add_virtual_links(pseudo_df,pseudo_G,start_node:int,end_nodes:list):

    '''
    Adds directed virtual links with length 0 needed to perform routing on the pseudo-dual graph network graph.
    
    Notes:
        network_df must have a source and target column with those names
        psudeo_df must have two columns for each source and target link
            for the source link: source_A and source_B
            for the target link: target_A and target_B
        run remove_virtual links afterwards to remove these virtual links
    '''    

    #grab all psuedo graph edges that contain the starting node in the SOURCE_A column (going away from starting node)
    starting_set = pseudo_df.loc[pseudo_df['source_A'] == start_node,['source_A','source']]
    starting_set.columns = ['source','target']

    #grab all psuedo graph edges that contain the starting node in the TARGET column (going towards the starting node)
    ending_set = pseudo_df.loc[pseudo_df['target_B'].isin(set(end_nodes)),['target','target_B']]
    ending_set.columns = ['source','target']
    
    virtual_edges = pd.concat([starting_set,ending_set],ignore_index=True)
    
    #add virtual edge
    for row in virtual_edges[['source','target']].itertuples(index=False):
        pseudo_G.add_weighted_edges_from([(row[0],row[1],0)],weight='weight') 

    return pseudo_G, virtual_edges

def remove_virtual_edges(pseudo_G,virtual_edges):
    '''
    Parameters
    ----------
    pseudo_G : networkx graph
        network graph containing the virtual edges
    virtual_edges : dataframe
        dataframe containing the virtual edges that need to be removed

    Returns
    -------
    pseudo_G : networkx graph
        network graph without the virtual edges

    '''
    for row in virtual_edges.itertuples(index=False):
        pseudo_G.remove_edges_from([(row[0],row[1])])
        
    return pseudo_G


def objective_function(betas,links,G,ods,trips_df,matched_traces,durations):
    
    start_time = time.perf_counter()
    
    #beta coefficients
    betas = np.asarray(betas)

    #use beta coefficients to calculate link costs
    attribute_cost_x = betas[0]*links['not_infra_x']+betas[1]*links['twolanes_x']+betas[2]*links['here_>30mph_x']
    attribute_cost_y = betas[0]*links['not_infra_y']+betas[1]*links['twolanes_y']+betas[2]*links['here_>30mph_y']
    links['cost_x'] = links['length_ft']*(1+attribute_cost_x) + 9000 #large number to keep value positive
    links['cost_y'] = links['length_ft']*(1+attribute_cost_y) + 9000 #large number to keep value positive

    #use beta coefficient to calculate turn cost
    base_turn_cost = 30 # from Lowry et al 2016 DOI: http://dx.doi.org/10.1016/j.tra.2016.02.003
    turn_costs = {
        'left': betas[3] * base_turn_cost,
        'right': betas[4] * base_turn_cost,
        'straight': betas[5] * base_turn_cost
    }
    turn_costs = links['turn_type'].map(turn_costs)

    #add everything together
    links['total_cost'] = links['cost_x'] + links['cost_y'] + turn_costs

    #turn into dict (test this)
    costs = dict(zip(list(zip(links['source'],links['target'])),links['total_cost']))

    #update edge weights
    nx.set_edge_attributes(G,values=costs,name='weight')
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')
    sources = list(set([x[0] for x in ods]))
    for source in sources:
        targets = list((set([x[1] for x in ods if x[0] == source])))
        #add virtual links
        G, virtual_edges = add_virtual_links(links,G,source,targets)
        #perform shortest path routing for all target nodes from source node (from one to all until target node has been visited)
        for target in targets:  

            length, node_list = nx.single_source_dijkstra(G,source,target,weight='weight')
            edge_list = [(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
            shortest_paths[(source,target)] = {'edge_list':edge_list,'length':length}
        
        #remove virtual links
        G = remove_virtual_edges(G,virtual_edges)


    # #calculate exact overlap
    # all_overlap = 0
    # for idx, row in trips_df.iterrows():

    #     try:

    #         modeled_edges = shortest_paths[row['od']]['edge_list']     
    
    #         chosen_edges = matched_traces[row['tripid']]['edges']
    #         chosen_edges = [(int(link1),int(link2)) for link1,link2 in chosen_edges]
            
    #         chosen_and_shortest = set(chosen_edges) & set(modeled_edges)
            
    #         overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
    #         all_overlap += overlap_length
        
    #     except:
    #         continue

    #calculate approximate overlap
    all_overlap = 0
    for idx, row in trips_df.iterrows():

        #need these to be gdfs
        modeled_edges = shortest_paths[row['od']]['edge_list']
        modeled_edges.geometry = modeled_edges.buffer(buffer_ft)

        chosen_edges = matched_traces[row['tripid']]['matched_trip']
        chosen_edges['original_length'] = chosen_edges.length
        
        overlapping = gpd.overlay(chosen_edges)
        
        overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
        all_overlap += overlap_length
        

            

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

#add turns
psuedo_df, psuedo_G = create_pseudo_dual_graph(links)

#create edge_dict for quicker lookup
links['linkid'] = list(zip(links['source'],links['target']))
edge_dict = links.set_index('linkid').to_dict('index')

#condense
links = links[['linkid','source','target','name','bearing','weight','geometry']]

#get midpoint of all links
links.geometry = links.geometry.centroid

#create pseudo dual graph
df_line, psuedo_G = create_pseudo_dual_graph(links)




links['length_ft'] = links.length




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
    bounds = [[-5,5],[-5,5],[-5,5]]
    x = minimize(objective_function, bounds, args=(links,G,ods,trips_df,matched_traces,durations), method='pso')
    end = time.time()
    print(f'Took {(end-start)/60/60} hours')
    results[segment_filepath] = (x.x,x.fun)

#%%
timestr = time.strftime("%Y-%m-%d-%H-%M")
with (fp/f"calibration_results_{timestr}.pkl").open('wb') as fh:
    pickle.dump(results,fh)


new_results = {key.parts[-1].split('.csv')[0]:items for key, items in results.items()}
new_results = pd.DataFrame.from_dict(new_results,orient='index',columns=['coefs','overlap'])
new_results[['not_beltline','not_infra','2lanes','30mph']] = new_results['coefs'].apply(pd.Series)