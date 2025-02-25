from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from models import RCNNWL
from ioutils import get_all_batch, get_feature_batch
import math
import sys
import random
from optparse import OptionParser
from rdkit import Chem, RDLogger

from utils.nn import linearND

RDLogger.DisableLog('rdApp.*')

# 定义常量
NK = 20
NK0 = 10
INVALID_BOND = -1

# 解析命令行参数
parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default="../data/train.txt")
parser.add_option("-m", "--save_dir", dest="save_path", default="../checkpoints")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
opts, args = parser.parse_args()

# 转换为整数
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)

# 数据预处理函数
smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)

# 定义模型
model = RCNNWL(hidden_size=hidden_size, depth=depth)

# 打印模型参数，确保模型中有可训练的参数
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"Name: {name}, Trainable: {param.requires_grad}, Size: {param.size()}")

# 检查模型是否有可训练的参数
if not list(model.parameters()):
    raise ValueError("Model has no trainable parameters. Please check the model definition.")

# 定义优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 定义数据集类
class ReactionDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                r, e = line.strip().split()
                self.data.append((r, e))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r, e = self.data[idx]
        reactants = r.split('>')[0]
        edits = e
        return reactants, edits

# 数据加载器的 collate_fn
def collate_fn(batch):
    src_batch, edit_batch = zip(*batch)
    src_tuple = smiles2graph_batch(src_batch)
    cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
    return (*src_tuple, cur_label, cur_bin, sp_label)

# 创建数据加载器
train_dataset = ReactionDataset(opts.train_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 训练循环
def train(model, train_loader, optimizer, scheduler, max_norm):
    model.train()
    total_loss = 0
    total_acc = 0
    total_err = 0
    total_gnorm = 0
    it = 0

    for batch in train_loader:
        it += 1
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, label, binary, sp_label = batch

        # 将 NumPy 数组转换为 PyTorch 张量
        input_atom = torch.tensor(input_atom, dtype=torch.float32)
        input_bond = torch.tensor(input_bond, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        binary = torch.tensor(binary, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        atom_graph = torch.tensor(atom_graph, dtype=torch.long)
        bond_graph = torch.tensor(bond_graph, dtype=torch.long)
        num_nbs = torch.tensor(num_nbs, dtype=torch.long)

        optimizer.zero_grad()

        # 前向传播
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

        flat_label = label.view(-1)
        bond_mask = (flat_label != INVALID_BOND).float()
        flat_score = score.view(-1)

        loss = criterion(flat_score, flat_label)
        loss = torch.sum(loss * bond_mask)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_gnorm += grad_norm

        # 计算准确率
        for i in range(batch_size):
            pre = 0
            for j in range(NK):
                if topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre:
                total_err += 1
            pre = 0
            for j in range(NK0):
                if topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre:
                total_acc += 1

        if it % 50 == 0:
            print(f"Iter: {it}, Loss: {total_loss / it:.4f}, Acc@10: {total_acc / (50 * batch_size):.4f}, Acc@20: {total_err / (50 * batch_size):.4f}, Grad Norm: {total_gnorm / 50:.2f}")
            total_acc, total_err, total_gnorm = 0, 0, 0

        if it % 10000 == 0:
            torch.save(model.state_dict(), f"{opts.save_path}/model.ckpt")
            print("Model saved!")

    torch.save(model.state_dict(), f"{opts.save_path}/model.final")
    print("Training complete. Model saved.")

# 主函数
if __name__ == "__main__":
    train(model, train_loader, optimizer, scheduler, max_norm)