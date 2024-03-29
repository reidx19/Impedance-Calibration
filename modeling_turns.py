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
import numpy as np
import pyproj

warnings.filterwarnings("ignore")

def make_multidigraph(network_df, source='source', target='target', linkid ='linkid', oneway='oneway', fwd_azimuth='fwd_azimuth', bck_azimuth='bck_azimuth'):
    MDG = nx.MultiDiGraph()  # Create a MultiDiGraph
    #itertuples used to maintain the type
    for row in network_df[[source, target, linkid, oneway, fwd_azimuth, bck_azimuth]].itertuples(index=False):
        edge_data = {linkid: row[2],'reverse_link': False, 'azimuth': row[4]}
        MDG.add_edge(row[0], row[1], **edge_data)  # Add edge with linkid attribute
        #add reverse link if oneway is not true
        if row[3] == False:
            edge_data['reverse_link'] = True 
            #reverse the azimuth
            edge_data['azimuth'] = row[5]
            MDG.add_edge(row[1], row[0], **edge_data)
    return MDG

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

    #grab all pseudo graph edges that contain the starting node in the SOURCE_A column (going away from starting node)
    starting_set = pseudo_df.loc[pseudo_df['source_A'] == start_node,['source_A','source']].drop_duplicates()
    starting_set.columns = ['source','target']

    #grab all pseudo graph edges that contain the starting node in the TARGET column (going towards the starting node)
    ending_set = pseudo_df.loc[pseudo_df['target_B'].isin(set(end_nodes)),['target','target_B']].drop_duplicates()
    ending_set.columns = ['source','target']
    
    virtual_edges = pd.concat([starting_set,ending_set],ignore_index=True)
    
    #add virtual edge
    for row in virtual_edges[['source','target']].itertuples(index=False):
        edge_data = {'weight':0}
        pseudo_G.add_edge(row[0],row[1],**edge_data)
    return pseudo_G, virtual_edges

def remove_virtual_edges(pseudo_G,virtual_edges):
    '''
    Reverses add_virtual_links
    '''
    for row in virtual_edges.itertuples(index=False):
        pseudo_G.remove_edge(row[0],row[1])
    return pseudo_G

def find_azimuth(row):
    coords = np.array(row.geometry.coords)
    lat1 = coords[0,1]
    lat2 = coords[-1,1]
    lon1 = coords[0,0]
    lon2 = coords[-1,0]

    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth,back_azimuth,distance = geodesic.inv(lon1, lat1, lon2, lat2)

    # print('Forward Azimuth:',fwd_azimuth)
    # print('Back Azimuth:',back_azimuth)
    # print('Distance:',distance)
    return pd.Series([np.round(fwd_azimuth,1) % 360, np.round(back_azimuth,1) % 360],index=['fwd_azimuth','bck_azimuth'])

def create_pseudo_dual_graph(edges,source_col,target_col,linkid_col,oneway_col):
    
    #simplify column names and remove excess variables
    edges.rename(columns={source_col:'source',target_col:'target',linkid_col:'linkid',oneway_col:'oneway'},inplace=True)
    edges = edges[['source','target','linkid','oneway','geometry']]
    
    #re-calculate azimuth (now azimuth)
    prev_crs = edges.crs
    edges.to_crs('epsg:4326',inplace=True)
    edges[['fwd_azimuth','bck_azimuth']] = edges.apply(lambda row: find_azimuth(row), axis=1)
    edges.to_crs(prev_crs,inplace=True)
    #edges['azimuth'] = edges.apply(lambda row: add_azimuth(row),axis=1)

    #turn into directed graph network wiht multiple edges
    G = make_multidigraph(edges)
    df_edges = nx.to_pandas_edgelist(G)

    #use networkx line graph function to create pseudo dual graph
    G_line = nx.line_graph(G)
    df_line = nx.to_pandas_edgelist(G_line)

    #get expanded tuples to columns for exporting purposes
    df_line[['source_A','source_B','source_Z']] = pd.DataFrame(df_line['source'].tolist(), index=df_line.index)
    df_line[['target_A','target_B','target_Z']] = pd.DataFrame(df_line['target'].tolist(), index=df_line.index)
    
    #drop the duplicate edges (these are addressed in the merge step)
    #line_graph doesn't carry over the linkid, so it resets multi-edges to 0 or 1
    df_line.drop(columns=['source','target','source_Z','target_Z'],inplace=True)
    df_line.drop_duplicates(inplace=True)

    #merge df_edges and df_line to add linkid and reverse_link keys back in
    df_line = df_line.merge(df_edges,left_on=['source_A','source_B'],right_on=['source','target'])
    df_line.rename(columns={'linkid':'source_linkid',
                            'reverse_link':'source_reverse_link',
                            'azimuth':'source_azimuth'},inplace=True)
    df_line.drop(columns=['source','target'],inplace=True)
    
    df_line = df_line.merge(df_edges,left_on=['target_A','target_B'],right_on=['source','target'])
    df_line.rename(columns={'linkid':'target_linkid',
                            'reverse_link':'target_reverse_link',
                            'azimuth':'target_azimuth'},inplace=True)
    df_line.drop(columns=['source','target'],inplace=True)
    
    #remove u-turns
    u_turn = (df_line['source_A'] == df_line['target_B']) & (df_line['source_B'] == df_line['target_A'])
    df_line = df_line[-u_turn]
    
    #change in azimuth
    df_line['azimuth_change'] = (df_line['target_azimuth'] - df_line['source_azimuth']) % 360
    
    #angle here
    '''
    straight < 30 or > 330
    right >= 30 and <= 150
    backwards > 150 and less than 210
    left >= 210 and <= 270 
    
    '''
    straight = (df_line['azimuth_change'] > 330) | (df_line['azimuth_change'] < 30) 
    right = (df_line['azimuth_change'] >= 30) & (df_line['azimuth_change'] <= 150)
    backwards = (df_line['azimuth_change'] > 150) & (df_line['azimuth_change'] < 210)
    left = (df_line['azimuth_change'] >= 210) & (df_line['azimuth_change'] <= 330)
    
    df_line.loc[straight,'turn_type'] = 'straight'
    df_line.loc[right,'turn_type'] = 'right'
    df_line.loc[backwards,'turn_type'] = 'uturn'
    df_line.loc[left,'turn_type'] = 'left'

    #create new source and target columns
    df_line['source'] = tuple(zip(df_line['source_A'],df_line['source_B']))
    df_line['target'] = tuple(zip(df_line['target_A'],df_line['target_B']))

    #remove duplicate edges (duplicates still retained in df_edges and df_line)
    pseudo_df = df_line[['source','target']].drop_duplicates()
    
    #pseudo graph too
    # not sure how to have shortest path algorithm report back the currect multigraph result
    # instead we use pseudo_df to know which link pairs had the lowest cost
    pseudo_G = nx.DiGraph()
    for row in pseudo_df[['source','target']].itertuples(index=False):
        edge_data = {'weight':1}
        pseudo_G.add_edge(row[0], row[1],**edge_data)

    return df_edges, df_line, pseudo_G






# def add_azimuth(row):
#     lat1 = row['geometry'].coords.xy[0][1]
#     lat2 = row['geometry'].coords.xy[-1][1]
#     lon1 = row['geometry'].coords.xy[0][0]
#     lon2 = row['geometry'].coords.xy[-1][0]

#     azimuth = calculate_azimuth(lat1,lon1,lat2,lon2)
    
#     return azimuth

# #from osmnx
# #it asks for coordinates in decimal degrees but returns all barings as 114 degrees?
# def calculate_azimuth(lat1, lon1, lat2, lon2):
#     """
#     Calculate the compass azimuth(s) between pairs of lat-lon points.

#     Vectorized function to calculate initial azimuths between two points'
#     coordinates or between arrays of points' coordinates. Expects coordinates
#     in decimal degrees. Bearing represents the clockwise angle in degrees
#     between north and the geodesic line from (lat1, lon1) to (lat2, lon2).

#     Parameters
#     ----------
#     lat1 : float or numpy.array of float
#         first point's latitude coordinate
#     lon1 : float or numpy.array of float
#         first point's longitude coordinate
#     lat2 : float or numpy.array of float
#         second point's latitude coordinate
#     lon2 : float or numpy.array of float
#         second point's longitude coordinate

#     Returns
#     -------
#     azimuth : float or numpy.array of float
#         the azimuth(s) in decimal degrees
#     """
#     # get the latitudes and the difference in longitudes, all in radians
#     lat1 = np.radians(lat1)
#     lat2 = np.radians(lat2)
#     delta_lon = np.radians(lon2 - lon1)

#     # calculate initial azimuth from -180 degrees to +180 degrees
#     y = np.sin(delta_lon) * np.cos(lat2)
#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
#     initial_azimuth = np.degrees(np.arctan2(y, x))

#     # normalize to 0-360 degrees to get compass azimuth
#     return initial_azimuth % 360




# #%% create pseudo graph

# #import links

# #create pseudo graph
# pseudo_edges, pseudo_G = create_pseudo_dual_graph(edges)



# #%% for routing


# pseudo_G, virtual_edges = add_virtual_links(edges, df_line, pseudo_G, start_node, end_nodes, 'weight')


# #calculate total cost
# df_line['turn_cost'] = df_line['turn_type'].map(turn_costs)
# (df_line['weight_turns'] = df_line['weight_x'] + df_line['weight_y']) * df_line['turn_cost']

# df_line['linkid'] = list(zip(df_line['source'],df_line['target']))
# turn_dict = df_line.set_index('linkid').to_dict('index')

# pseudo_G = make_graph(df_line)
# pseudo_G_turn_costs = make_graph(df_line,weight='weight_turns')

# #dijkstra takes a single source and finds path to all possible target nodes
# start_node = 69531971
# end_nodes = [69279745]

# #make tuple columns for easier matching
# df_line['source'] = list(zip(df_line['source_A'],df_line['source_B']))
# df_line['target'] = list(zip(df_line['target_A'],df_line['target_B']))




# #peform routing
# impedances, paths = nx.single_source_dijkstra(pseudo_G,start_node,weight="weight")

# #remove virtual edges
# pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

# #%% analyze

# end_node = end_nodes[0]
    
# path = paths[end_node]

# #return list of edges without the virtual links
# edge_list = path[1:-1]

# #return list of turns taken
# turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

# #get edge geometry
# edge_gdf = [edge_dict.get(id,0) for id in edge_list]

# #get turn geometry
# turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
# #turn edges into gdf
# edge_gdf = pd.DataFrame.from_records(edge_gdf)
# turn_gdf = pd.DataFrame.from_records(turn_gdf)

# #turn into gdf
# edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
# turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

# #export for examination
# edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
# turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


# #%% for turn costs 

# #add virtual edges
# pseudo_G, virtual_edges = add_virtual_links(edges, df_line, pseudo_G_turn_costs, start_node, end_nodes, 'weight')

# #peform routing
# impedances, paths = nx.single_source_dijkstra(pseudo_G_turn_costs,start_node,weight="weight")

# #remove virtual edges
# pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

# #%% analyze

# end_node = end_nodes[0]
    
# path = paths[end_node]

# #return list of edges without the virtual links
# edge_list = path[1:-1]

# #return list of turns taken
# turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

# #get edge geometry
# edge_gdf = [edge_dict.get(id,0) for id in edge_list]

# #get turn geometry
# turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
# #turn edges into gdf
# edge_gdf = pd.DataFrame.from_records(edge_gdf)
# turn_gdf = pd.DataFrame.from_records(turn_gdf)

# #turn into gdf
# edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
# turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

# #export for examination
# edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
# turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


