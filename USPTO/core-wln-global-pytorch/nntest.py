import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from models import RCNNWL, linearND
from ioutils import get_all_batch, get_feature_batch, INVALID_BOND
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

NK = 40

parser = OptionParser()
parser.add_option("-t", "--test", dest="train_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
opts, args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)

smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)

# Define the model
model = RCNNWL(hidden_size=hidden_size, depth=depth)
model.load_state_dict(torch.load(opts.model_path))
model.eval()

# Placeholder for input tensors
_input_atom = torch.zeros(batch_size, 1, adim)
_input_bond = torch.zeros(batch_size, 1, bdim)
_atom_graph = torch.zeros(batch_size, 1, max_nb, 2, dtype=torch.int32)
_bond_graph = torch.zeros(batch_size, 1, max_nb, 2, dtype=torch.int32)
_num_nbs = torch.zeros(batch_size, 1, dtype=torch.int32)
_node_mask = torch.zeros(batch_size, 1)
_label = torch.zeros(batch_size, 1)
_binary = torch.zeros(batch_size, 1, 1, bdim)

# Define the forward pass
def forward_pass(input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, label, binary):
    atom_hiddens, _ = model(input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, batch_size)
    atom_hiddens1 = atom_hiddens.view(batch_size, 1, -1, hidden_size)
    atom_hiddens2 = atom_hiddens.view(batch_size, -1, 1, hidden_size)
    atom_pair = atom_hiddens1 + atom_hiddens2

    att_hidden = F.relu(linearND(atom_pair, hidden_size) + linearND(binary, hidden_size))
    att_score = linearND(att_hidden, 1)
    att_score = torch.sigmoid(att_score)
    att_context = att_score * atom_hiddens1
    att_context = torch.sum(att_context, dim=2)

    att_context1 = att_context.view(batch_size, 1, -1, hidden_size)
    att_context2 = att_context.view(batch_size, -1, 1, hidden_size)
    att_pair = att_context1 + att_context2

    pair_hidden = linearND(atom_pair, hidden_size) + linearND(binary, hidden_size) + linearND(att_pair, hidden_size)
    pair_hidden = F.relu(pair_hidden)
    pair_hidden = pair_hidden.view(batch_size, -1, hidden_size)

    score = linearND(pair_hidden, 1)
    score = score.squeeze(2)
    bmask = torch.where(label == INVALID_BOND, torch.tensor(10000.0), torch.tensor(0.0))
    _, topk = torch.topk(score - bmask, k=NK)
    return topk

# Data reading and processing
queue = Queue()

def read_data(path, coord):
    data = []
    with open(path, 'r') as f:
        for line in f:
            r, e = line.strip("\r\n ").split()
            data.append((r, e))

    for it in range(0, len(data), batch_size):
        src_batch, edit_batch = [], []
        for i in range(batch_size):
            react, _, p = data[it][0].split('>')
            src_batch.append(react)
            edits = data[it][1]
            edit_batch.append(edits)
            it = (it + 1) % len(data)

            pmol = Chem.MolFromSmiles(p)
            patoms = set([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
            ratoms = []
            for x in react.split('.'):
                xmol = Chem.MolFromSmiles(x)
                xatoms = [atom.GetAtomMapNum() for atom in xmol.GetAtoms()]
                if len(set(xatoms) & patoms) > 0:
                    ratoms.extend(xatoms)
            queue.put(ratoms)

        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        feed_map = {x: y for x, y in zip([_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask], src_tuple)}
        feed_map.update({_label: cur_label, _binary: cur_bin})
        topk = forward_pass(**feed_map)

        for i in range(batch_size):
            ratoms = queue.get()
            for j in range(NK):
                k = topk[i, j]
                x = k // cur_dim + 1
                y = k % cur_dim + 1
                if x < y and x in ratoms and y in ratoms:
                    print("%d-%d" % (x, y)),
        print()

# Start data reading thread
coord = threading.Thread(target=read_data, args=(opts.train_path, None))
coord.start()

# Main loop
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping threads...")
finally:
    coord.join()