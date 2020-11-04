import json
import networkx as nx 
import torch
from transformers import *
from pyHGT.utils import *
from networkx.readwrite import json_graph
from pyHGT.data import *
import pymongo
import json
import argparse

parser = argparse.ArgumentParser(description='Preprocess OAG (CS/Med/All) Data')
'''
    Dataset arguments
'''
parser.add_argument('--input_graph', type=str, default='../../swiss_uni/split_data/result_pale-G.json',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='converted_graph.pk',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=str, default='0',
                    help='Avaiable GPU ID')
args = parser.parse_args()


client = pymongo.MongoClient("mongodb://localhost:2710/")
swiss_db = client["swiss"]
patent_collect = swiss_db['patent']
indeed_collect = swiss_db['indeed']

graph_json = json.load(open(args.input_graph, 'r'))
nx_graph = json_graph.node_link_graph(graph_json)

graph = Graph()
edg = graph.edge_list

# type: company, tech, indeed, mag, patent = list(range(5))
type_dict = {i: [] for i in range(5)} # dict to save nodes in the same type (key is type of node)
for node in nx_graph.nodes(data = True): 
    type_dict[node[1]['label'][0]].append(node[0])

id2idx = {node:j for i in range(5) for (j,node) in enumerate(type_dict[i])} # change node_id in each type to range start from 0 ...
node2type = {}
for (j,z) in type_dict.items():
    for i in z:
        node2type[i] = j

### Add edges:
checked = []
for edge in nx_graph.edges():
    checked.append((edge[0], edge[1]))
    n0 = graph_json['nodes'][edge[0]]
    n0 = {'id': n0['id'], 'type': n0['label'][0], 'time':2020}
    n1 = graph_json['nodes'][edge[1]]
    n1 = {'id': n1['id'], 'type': n1['label'][0], 'time':2020}
    graph.edge_list[n1['type']][n0['type']][0][id2idx[n1['id']]][id2idx[n0['id']]] = 2020
    graph.edge_list[n0['type']][n1['type']][0][id2idx[n0['id']]][id2idx[n1['id']]] = 2020

### Add node_features:

# Embed indeed/patent:
if args.cuda == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device("cuda:" + args.cuda)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased',
                                    output_hidden_states=True,
                                    output_attentions=True).to(device)
embs = np.empty((len(type_dict[3]), 768))
for node in type_dict[3]:
    text = patent_collect.find({'_id':graph_json['nodes'][node]['content'][0]})[0]['abstract']
    input_ids = torch.tensor([tokenizer.encode(text)]).to(device)[:, :64]
    if len(input_ids[0]) < 4:
        continue
    all_hidden_states, all_attentions = model(input_ids)[-2:]
    rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
    embs[id2idx[node]] = rep.detach().cpu().numpy()
graph.node_feature[3] = embs

embs = np.empty((len(type_dict[2]), 768))
for node in type_dict[2]:
    text = indeed_collect.find({'_id':graph_json['nodes'][node]['content'][0]})[0]['jobDescription']
    input_ids = torch.tensor([tokenizer.encode(text)]).to(device)[:, :64]
    if len(input_ids[0]) < 4:
        continue
    all_hidden_states, all_attentions = model(input_ids)[-2:]
    rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
    embs[id2idx[node]] = rep.detach().cpu().numpy()
graph.node_feature[2] = embs

for _type in range(5):
    if _type in [0,1]:
        cv = graph.node_feature[2]
        i = []
        for s in graph.edge_list[_type][2][0]:
            for t in graph.edge_list[_type][2][0][s]:
                i += [[s,t]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(type_dict[_type]), len(type_dict[2]))))
        out = m.dot(cv)

        cv = graph.node_feature[3]
        i = []
        for s in graph.edge_list[_type][3][0]:
            for t in graph.edge_list[_type][3][0][s]:
                i += [[s,t]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(type_dict[_type]), len(type_dict[3]))))
        out += m.dot(cv)

        graph.node_feature[_type] = out 
    
    elif _type == 4:
        cv = graph.node_feature[0]
        i = []
        for s in graph.edge_list[_type][0][0]:
            for t in graph.edge_list[_type][0][0][s]:
                i += [[s,t]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(type_dict[_type]), len(type_dict[0]))))
        out = m.dot(cv)

        cv = graph.node_feature[1]
        i = []
        for s in graph.edge_list[_type][1][0]:
            for t in graph.edge_list[_type][1][0][s]:
                i += [[s,t]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(type_dict[_type]), len(type_dict[1]))))
        out += m.dot(cv)

        graph.node_feature[_type] = out 

dill.dump(graph, open(args.output_dir, 'wb'))