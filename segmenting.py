#https://keurfonluu.github.io/stochopy/api/optimize.html

from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import json

'''
This module is for performing k-means clustering using detour rate and/or speed on
cycleatlanta users

'''

fp = Path.home() / "Documents/GitHub/Impedance-Calibration"

#import processed trips
trips_df = pd.read_csv(fp/'trips_df_postmatch.csv')

#drop trips more than five miles
trips_df['chosen_length'] = trips_df['chosen_length'] / 5280
trips_df = trips_df[trips_df['chosen_length']<5]

trips_df['euclidean_distance'] = trips_df['euclidean_distance'] / 5280

removal_words = ['critical mass','mobile social','dikov ride']

#for now, throw out trips mentioning group rides and those with detour rate above 100 (twice the distance)

#import mapping
user_data_definitions = json.load(open(fp/'user_data_definition.json'))

#%% remove if same od within user id

trips_df = trips_df[-trips_df[['userid','od']].duplicated()]

#%%
def export_segments(column_name,categorical,trips_df,values_to_exclude,user_data_definitions):
    if categorical:
        trips_df[column_name] = trips_df[column_name].astype(str)
        trips_df[column_name] = trips_df[column_name].map(user_data_definitions[column_name])

    for value in trips_df[column_name].dropna().unique():
        if value in values_to_exclude:
            continue
        to_sample = trips_df[trips_df[column_name]==value]
        
        try:
            sample = to_sample.sample(200)
            sample.to_csv(fp/f'segments/{column_name}-{value}.csv',index=False)
        except:
            print(value,'did not have enough values')
            continue

    #trips_df.drop(columns=[column_name+'temp'],inplace=True)
    
export_segments('gender',True,trips_df,['no data'],user_data_definitions)
export_segments('ethnicity',True,trips_df,['no data'],user_data_definitions)
export_segments('age',True,trips_df,['no data'],user_data_definitions)
export_segments('income',True,trips_df,['no data'],user_data_definitions)
export_segments('trip_type',False,trips_df,['no data'],user_data_definitions)
export_segments('rider_type',False,trips_df,['no data'],user_data_definitions)

#%%
'''
K-means

Variables:
trip distance
detour %
dist to: work, home, school

aim for 3-4 clusters

on longer trips people are more likely to detour, for short trips directness prefereed?
casual riders are travelling shorter distances and may be more avoidant of certain roads


some of the really high detour trip are still loops
valid but need to have better detection for pauses


most of the data is just winding up in one cluster, so i need to think harder about what i am clustering/grouping on


'''
import matplotlib.pyplot as plt

#cluster using trip distance and detour %


fig, axis = plt.subplots(figsize =(10, 5))
bins = np.array([x for x in range(0, 300, 5)])
axis.hist(trips_df['detour_rate'], bins = bins)
plt.xlabel('Percent Detour')
plt.ylabel('Frequency')

# Calculate the median
median_value = np.median(trips_df['detour_rate'])

# Draw a vertical line at the median
plt.axvline(median_value, color='red', linestyle='dashed', linewidth=2, label=f'Median = {median_value}')

# Label the vertical line
#plt.text(median_value + 10, 20, f'Median = {median_value}', rotation=90, color='red')

# Displaying the graph
plt.legend()
plt.show()

#%%


#turn to array
X = np.asarray(trips_df[['detour_rate','chosen_length']])

# do clustering
kmeans = KMeans(n_clusters=3).fit(X)

trips_df['cluster_label'] = kmeans.labels_
results = pd.DataFrame(kmeans.cluster_centers_, columns = ['detour_rate','chosen_length'])
print(results)

# for cluster_label in trips_df['cluster_label'].dropna().unique():
#     to_sample = trips_df[trips_df['cluster_label']==cluster_label]
#     sample = to_sample.sample(50)
#     sample.to_csv(fp/f'segments/cluster_{cluster_label}.csv',index=False)



#cluter using euclidean distance to work/home/school too


'''
come back to, right now most of the data is just in one cluster

K-prototypes (accepts both numerical and catagorical)

Trip purpose
ethnicity
gender
income

'''







