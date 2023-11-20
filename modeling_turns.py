#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:13:02 2023

@author: tannerpassmore
"""
import networkx as nx
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, Point, LineString
import warnings
warnings.filterwarnings("ignore")

#%% functions

def make_graph(network_df,source='source',target='target',weight='weight'):
    G = nx.DiGraph() 
    for row in network_df[[source,target,weight]].itertuples(index=False):
        G.add_weighted_edges_from([(row[0],row[1],row[2])],weight='weight')
    return G 

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






#apply turn costs
df_line['turn_cost'] = df_line['turn_type'].map(turn_costs)
df_line['weight_turns'] = df_line['weight_x'] + df_line['weight_y'] + df_line['turn_cost']

df_line['linkid'] = list(zip(df_line['source'],df_line['target']))
turn_dict = df_line.set_index('linkid').to_dict('index')

psuedo_G = make_graph(df_line)
psuedo_G_turn_costs = make_graph(df_line,weight='weight_turns')

#dijkstra takes a single source and finds path to all possible target nodes
start_node = 69531971
end_nodes = [69279745]

#make tuple columns for easier matching
df_line['source'] = list(zip(df_line['source_A'],df_line['source_B']))
df_line['target'] = list(zip(df_line['target_A'],df_line['target_B']))


#%% without turn costs

#add virtual edges
pseudo_G, virtual_edges = add_virtual_links(edges, df_line, psuedo_G, start_node, end_nodes, 'weight')

#peform routing
impedances, paths = nx.single_source_dijkstra(psuedo_G,start_node,weight="weight")

#remove virtual edges
pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

#%% analyze

end_node = end_nodes[0]
    
path = paths[end_node]

#return list of edges without the virtual links
edge_list = path[1:-1]

#return list of turns taken
turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

#get edge geometry
edge_gdf = [edge_dict.get(id,0) for id in edge_list]

#get turn geometry
turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
#turn edges into gdf
edge_gdf = pd.DataFrame.from_records(edge_gdf)
turn_gdf = pd.DataFrame.from_records(turn_gdf)

#turn into gdf
edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

#export for examination
edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


#%% for turn costs 

#add virtual edges
pseudo_G, virtual_edges = add_virtual_links(edges, df_line, psuedo_G_turn_costs, start_node, end_nodes, 'weight')

#peform routing
impedances, paths = nx.single_source_dijkstra(psuedo_G_turn_costs,start_node,weight="weight")

#remove virtual edges
pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

#%% analyze

end_node = end_nodes[0]
    
path = paths[end_node]

#return list of edges without the virtual links
edge_list = path[1:-1]

#return list of turns taken
turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

#get edge geometry
edge_gdf = [edge_dict.get(id,0) for id in edge_list]

#get turn geometry
turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
#turn edges into gdf
edge_gdf = pd.DataFrame.from_records(edge_gdf)
turn_gdf = pd.DataFrame.from_records(turn_gdf)

#turn into gdf
edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

#export for examination
edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


