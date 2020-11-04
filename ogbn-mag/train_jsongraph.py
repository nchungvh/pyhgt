import argparse
from tqdm import tqdm
import sys
import random

from sklearn.metrics import f1_score
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


parser = argparse.ArgumentParser(description='Training GNN on ogbn-mag benchmark')



parser.add_argument('--data_dir', type=str, default='converted_graph.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./hgt_4layer',
                    help='The address for storing the trained models.')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=str, default='cpu',
                    help='Avaiable GPU ID')
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=512,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=520,
                    help='How many nodes to be sampled per layer per type')

parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of output nodes for training')  
parser.add_argument('--clip', type=int, default=1.0,
                    help='Gradient Norm Clipping') 
parser.add_argument('--field', type=str, default='com',
                    help='Field for get embeddings')

parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--use_RTE',   help='Whether to use RTE',     action='store_true')

args = parser.parse_args()
args_print(args)
if args.field == 'com':
    field = 0
elif args.field == 'tech':
    field = 1
def ogbn_sample(seed, samp_nodes):
    np.random.seed(seed)
    feature, times, edge_list, indxs, _ = sample_subgraph(graph, \
                inp = {field: np.concatenate([samp_nodes, [2020]*len(samp_nodes)]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    return node_feature, node_type, edge_time, edge_index, edge_type, indxs
    
def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), \
                            np.random.choice(target_nodes, args.batch_size, replace = False)]))
            jobs.append(p)
    return jobs

def triplet_loss(graph, feature, indxs, source = 0):
    loss = 0
    types = graph.get_types()
    count = 0
    times = 1000
    feat_dict = {}
    for _type in types:
        feat_dict[_type] = feature[count: count + len(indxs[_type])]
        count += len(indxs[_type])
    nei_types = [t for t in types if t in graph.edge_list[source].keys()]
    while times > 0:
        t1 = random.choice(nei_types)
        k = random.choice(indxs[t1])
        for s in indxs[source]:
            if random.uniform(0,1)> 0.5 and s in graph.edge_list[t1][source][0][k].keys():
                loss += sum(feat_dict[source][np.where(indxs[source] == s)[0][0]] * feat_dict[t1][np.where(indxs[t1] == k)[0][0]])
                break
        t2 = random.choice(types)
        s = random.choice(indxs[source])
        k = random.choice(indxs[t2])
        loss -=  5 * sum(feat_dict[source][np.where(indxs[source] == s)[0][0]] * feat_dict[t2][np.where(indxs[t2] == k)[0][0]])
        times -= 1
    return loss


graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
if args.cuda == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device("cuda:" + args.cuda)

target_nodes = np.arange(len(graph.node_feature[field]))
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[field][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
# classifier = Classifier(args.n_hid, graph.y.max()+1)

model = nn.Sequential(gnn).to(device)
print('Model #Params: %d' % get_n_params(model))
criterion = nn.NLLLoss()


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)

stats = []
res   = []
best_val   = 0
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)


for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    datas = [job.get() for job in jobs]
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    model.train()
    stat = []
    for node_feature, node_type, edge_time, edge_index, edge_type, indxs in datas:
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))

        train_loss = triplet_loss(graph, node_rep, indxs, source = field)

        optimizer.zero_grad() 
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_step += 1
        scheduler.step(train_step)

        stat += [[train_loss.item()]]
        del node_rep, train_loss
    stats += [stat]
    avgs = np.average(stat, axis=0)
    
    print('Epoch: %d LR: %.5f Train Loss: %.4f ' % \
         (epoch,  optimizer.param_groups[0]['lr'], avgs[0]))

print('Get embedding ....')
model.eval()
rest_nodes = target_nodes + 0
count = 1000
embs = torch.zeros(graph.node_feature[field].shape[0], 32)
while rest_nodes.shape[0] > 0 and count > 0:
    print('okkkk')
    if rest_nodes.shape[0] < args.batch_size:
        node_feature, node_type, edge_time, edge_index, edge_type, indxs \
         = ogbn_sample(randint(), rest_nodes)
    else:
        node_feature, node_type, edge_time, edge_index, edge_type, indxs \
         = ogbn_sample(randint(), np.random.choice(rest_nodes, args.batch_size, replace = False))

    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                            edge_time.to(device), edge_index.to(device), edge_type.to(device))


    rest_nodes = np.asarray([i for i in rest_nodes if i not in indxs[field]])  
    types = graph.get_types()
    count = 0
    feat_dict = {}
    for _type in types:
        feat_dict[_type] = node_rep[count: count + len(indxs[_type])]
        count += len(indxs[_type])  
    for i,feat in enumerate(indxs[field]):
        embs[feat] = feat_dict[field][i]
embs = embs.detach().cpu().numpy()
np.save('{}_embeddings.npy'.format(args.field), embs)    

