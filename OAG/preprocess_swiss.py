import json
import networkx as nx 
from networkx.readwrite import json_graph
from pyHGT.data import *

PATH = 'DBpedia-G.json'
graph_json = json.load(open(PATH, 'r'))
graph = json_graph.node_link_graph(graph_json)

graph = Graph()
checked = []
for edge in graph.edges():
    if (edge[0], edge[1]) in checked or (edge[1], edge[0]) in checked:
        continue
    checked.append((edge[0], edge[1]))
    n0 = graph_json['nodes'][edge[0]]
    n0 = {'id': n0['id'], 'type': n0['label'][0], 'time':2020}
    n1 = graph_json['nodes'][edge[1]]
    n1 = {'id': n1['id'], 'type': n1['label'][0], 'time':2020}
    if n0['type'] > n1['type']:
        rel = str(n0['type']) + str(n1['type'])
        graph.add_edge(n0, n1, time = 2020, relation_type = rel)
    else:
        rel = str(n1['type']) + str(n0['type'])
        graph.add_edge(n1, n0, time = 2020, relation_type = rel)
    