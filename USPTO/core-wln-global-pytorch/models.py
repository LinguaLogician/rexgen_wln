import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNNWL(nn.Module):
    def __init__(self, hidden_size, depth, training=True):
        super(RCNNWL, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.training = training

        # 添加可训练的层
        # self.atom_embedding = nn.Linear(128, hidden_size)  # 假设输入的原子特征维度为128
        self.atom_embedding = nn.Linear(82, hidden_size)  # 假设输入的原子特征维度为128
        self.nei_atom_linear = nn.Linear(hidden_size, hidden_size)
        self.nei_bond_linear = nn.Linear(hidden_size, hidden_size)
        self.self_atom_linear = nn.Linear(hidden_size, hidden_size)
        self.label_U1 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, batch_size, max_nb=10):
        atom_features = F.relu(self.atom_embedding(input_atom))  # 原子特征嵌入

        for i in range(self.depth):
            # 手动实现类似 tf.gather_nd 的索引逻辑
            batch_indices = torch.arange(batch_size, device=input_atom.device).view(-1, 1, 1).expand(-1, max_nb, 1)
            atom_indices = atom_graph[:, :, 0].unsqueeze(-1)  # 假设 atom_graph 是 [batch_size, max_nb, 2]
            bond_indices = bond_graph[:, :, 0].unsqueeze(-1)  # 假设 bond_graph 是 [batch_size, max_nb, 2]

            indices_atom = torch.cat([batch_indices, atom_indices], dim=-1).view(-1, 2)
            indices_bond = torch.cat([batch_indices, bond_indices], dim=-1).view(-1, 2)

            fatom_nei = input_atom[indices_atom[:, 0], indices_atom[:, 1]].view(batch_size, -1, max_nb,
                                                                                self.hidden_size)
            fbond_nei = input_bond[indices_bond[:, 0], indices_bond[:, 1]].view(batch_size, -1, max_nb,
                                                                                self.hidden_size)

            fatom_nei = self.nei_atom_linear(fatom_nei)
            fbond_nei = self.nei_bond_linear(fbond_nei)
            h_nei = fatom_nei * fbond_nei

            mask_nei = torch.unsqueeze(torch.sequence_mask(num_nbs.view(-1), max_nb, dtype=torch.float32), -1)
            f_nei = torch.sum(h_nei * mask_nei, dim=-2)

            f_self = self.self_atom_linear(atom_features)
            new_label = torch.cat([atom_features, f_nei], dim=-1)
            new_label = F.relu(self.label_U1(new_label))
            atom_features = new_label

        fp = torch.sum(atom_features, dim=1)
        return atom_features, fp