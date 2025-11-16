import math
import osmnx as ox
import pandas as pd
import numpy as np
import heapq
# import time
import matplotlib as mp
import matplotlib.colors as colors
import matplotlib.pyplot as plt

#####################################################################

def initialize_data(location):
    place = location
    G_initial = ox.graph.graph_from_place(place, network_type="drive")
    G = G_initial.reverse(copy=True) 
    #Reverse the graph so Djikstra's algorithm can be calculated from hospitals instead of to them

    ox.routing.add_edge_speeds(G, hwy_speeds=None, fallback=None, agg=np.mean)
    ox.routing.add_edge_travel_times(G)

    features = ox.features.features_from_place(place, {"amenity": "hospital"})

    feature_points = features.representative_point()
    nn = ox.distance.nearest_nodes(G, feature_points.x, feature_points.y)
    useful_tags = ["name", "emergency"]

    for node, feature in zip(nn, features[useful_tags].to_dict(orient="records")):
        feature = {k: v for k, v in feature.items() if pd.notna(v)}
        G.nodes[node].update({"hospital": feature})
        
    return G

print("intitializing data...", end="\r")    
G = initialize_data("New York, NY, USA")

#############################################################################
print("identifying hospital nodes...", end="\r")  

hospital_node_ids = []

for node in list(G.nodes):
    attr_name = 'hospitals'
    attr = set()
    
    G.nodes[node][attr_name] = attr
    
    if 'hospital' in G.nodes[node]:
        hospital_node_ids.append(node)

 

####################################################################################     

def dijkstra_to_hospital(graph, hospital_node):
    travel_times = {node: float('inf') for node in graph.nodes}
    travel_times[hospital_node] = 0
    pq = [(0, hospital_node)]

    while pq:
        current_travel_time, current_node = heapq.heappop(pq)

        if current_travel_time > travel_times[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            min_edge_time = min(attr['travel_time'] for attr in edge_data.values())

            travel_time = current_travel_time + min_edge_time

            if travel_time < travel_times[neighbor]:
                travel_times[neighbor] = travel_time
                heapq.heappush(pq, (travel_time, neighbor))

    return travel_times

####################################################################################
print("calculating distance...", end="\r")  

for hospital_node in hospital_node_ids:
    distance_to_nodes = dijkstra_to_hospital(G, hospital_node)
    for node, travel_time in distance_to_nodes.items():
        G.nodes[node]['hospitals'].add(travel_time)
        
for node in G.nodes:
    hospitals = sorted(G.nodes[node]['hospitals'])
    if hospitals[0] == float('inf'):
        G.nodes[node]['first_hospital'] = np.nan #unconected nodes get high scores
        G.nodes[node]['second_hospital'] = np.nan #inf would break cmap, so avoid
        continue
    G.nodes[node]['first_hospital'] = hospitals[0]
    G.nodes[node]['second_hospital'] = hospitals[1]

###############################################################################
for edge in G.edges:
    first = max(G.nodes[edge[0]]['first_hospital'], G.nodes[edge[1]]['first_hospital'])
    second = max(G.nodes[edge[0]]['second_hospital'], G.nodes[edge[1]]['second_hospital'])
    G.edges[edge]['first_hospital'] = first
    G.edges[edge]['second_hospital'] = second

###############################################################################

closest_hospital_total = 0
second_hospital_total = 0
for node in G.nodes:
    if not math.isnan(G.nodes[node]['first_hospital']):
        closest_hospital_total += G.nodes[node]['first_hospital']
    if not math.isnan(G.nodes[node]['second_hospital']):
        second_hospital_total += G.nodes[node]['second_hospital']

average_first = round(closest_hospital_total / len(G.nodes), 2)
average_second = round(second_hospital_total / len(G.nodes), 2)

def to_time(time):
    minutes = time // 60
    seconds = time - (minutes*60)
    return f"{int(minutes)} minutes {int(seconds)} seconds"

#####################################################################################

edge_colors = []
edge_colors2 = []


node_colors = []
node_colors2 = []


optimal_time = 0
max_time = 600    

# Normalize for colormap
norm = colors.Normalize(vmin=0, vmax=max_time)
cmap = mp.colormaps['RdYlGn_r']

for edge in G.edges:
    color = cmap(norm(G.edges[edge]['first_hospital']))
    color2 = cmap(norm(G.edges[edge]['second_hospital']))
    
    edge_colors.append(color)
    edge_colors2.append(color2)
    
for node in G.nodes:
    color = cmap(norm(G.nodes[node]['first_hospital']))
    color2 = cmap(norm(G.nodes[node]['second_hospital']))
    
    node_colors.append(color)
    node_colors2.append(color2)

print("rendering map of closest hospital...", end="\r")   
fig, ax = ox.plot.plot_graph(
    G, 
    figsize=(20, 20), 
    node_size=2, 
    node_color= node_colors, 
    edge_color= edge_colors,
    show=False,
    close=False
)

sm = plt.cm.ScalarMappable(cmap= cmap, norm= norm)
cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label("Travel Time to Nearest Hospital", fontsize=15)

ax.set_title("Distance to Nearest Hospital - New York City", fontsize=20)

plt.show()

print("rendering map of second-closest hospital...", end="\r")  
fig2, ax2 = ox.plot.plot_graph(
    G, 
    figsize=(20, 20), 
    node_size=2, 
    node_color= node_colors2, 
    edge_color= edge_colors2,
    show=False,
    close=False
)

ax2.set_title("Distance to Second Nearest Hospital - New York City", fontsize=20)
cbar2 = fig.colorbar(sm, ax=ax2, shrink=0.5)
cbar2.set_label("Travel Time to Second Nearest Hospital", fontsize=15)

plt.show()


print(" ", end="\r")
print("Average travel time to nearest hospital: ", to_time(average_first))
print("Average travel time to second hospital: ", to_time(average_second))