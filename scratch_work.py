# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:14:36 2023

@author: tpassmore6
"""

import geopandas

gdf = gpd.read_file(r'C:/Users/tpassmore6/Downloads/cleaned_trips/matched_traces/6868.gpkg',layer='points')
gdf['datetime'] = pd.to_datetime(gdf['datetime'])
gdf['time_diff'] = gdf['datetime'].diff()
gdf['time_diff'].max()


'''
need to be able to detect group rides

'''