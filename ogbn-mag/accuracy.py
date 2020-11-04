import sys
import numpy as np
label_path = sys.argv[1]
meta_path = sys.argv[2]
emb_path = sys.argv[3]
with open(label_path, 'r') as f:
    labels = f.readlines()
    labels = [l.replace('\n', '') for l in labels]
    labels = {label.split('\t')[0]: label.split('\t')[1].split('|') for label in labels}
with open(meta_path, 'r') as f:
    metas = f.readlines()[1:]
    metas = [l.replace('\n', '') for l in metas]
    metas = {meta.split('\t')[1].replace('/',''): i for (i,meta) in enumerate(metas)}
print(len(metas))
metas = {i:j for (i,j) in metas.items() if i in labels.keys()}
full_embs = []
print(len(metas))

with open(emb_path, 'r') as f:
    embs = f.readlines()[1:]
    embs = [l.replace('\n', '') for l in embs]
    for emb in embs:
        emb = emb.split()
        emb = [float(e) for e in emb]
        full_embs.append(emb)
full_embs = np.asarray(full_embs)
import pdb
pdb.set_trace()