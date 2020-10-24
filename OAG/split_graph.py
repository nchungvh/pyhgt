import json
import networkx as nx 
from networkx.readwrite import json_graph
from pyHGT.data import *

PATH = 'DBpedia-G.json'
graph_json = json.load(open(PATH, 'r'))
graph = json_graph.node_link_graph(graph_json)
mask = json.load(open('mask_graph_split.json','r'))
subs = {i:[] for i in range(5)}
for idx, cluster in enumerate(mask):
    subs[cluster].append(idx)
for key in subs.keys():
    sub_graph = graph.subgraph(subs[key])
    res = json_graph.node_link_data(sub_graph)
    with open("subgraph_{}_G.json".format(key), 'w') as outfile:
        json.dump(res, outfile)
