import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Batch Normalization
def batch_normalization(x, scope, decay=0.999, eps=1e-6, training=True):
    return nn.BatchNorm1d(x.shape[-1])(x)

# Batch Normalization with Mask
def batch_normalization_with_mask(x, mask, scope, decay=0.999, eps=1e-6, training=True):
    # Custom implementation to handle mask
    mean = (x * mask).sum(dim=0) / mask.sum(dim=0)
    var = ((x - mean) * mask).pow(2).sum(dim=0) / mask.sum(dim=0)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm

# Linear Layer
def linear(input_, output_size, scope=None, reuse=False, init_bias=0.0):
    return nn.Linear(input_.shape[-1], output_size, bias=init_bias is not None)(input_)

# Linear Layer for ND Tensors
def linearND(input_, output_size, scope=None, reuse=False, init_bias=0.0):
    original_shape = input_.shape
    reshaped_input = input_.view(-1, original_shape[-1])
    linear_output = nn.Linear(reshaped_input.shape[-1], output_size, bias=init_bias is not None)(reshaped_input)
    return linear_output.view(*original_shape[:-1], output_size)

# Lookup Table
def lookup_table(input_, vocab_size, output_size, scope):
    return nn.Embedding(vocab_size, output_size)(input_)

# Sparse Linear Layer
def sparse_linear(input_, input_size, output_size, scope, init_bias=0.0):
    return nn.Linear(input_size, output_size, bias=init_bias is not None)(input_)

# CSR Matrix to PyTorch Tensor
def CSR2TF(x):
    indices = torch.tensor([[0, 0]])  # Avoid empty matrix
    values = torch.tensor([0.0])
    for i in range(x.shape[0]):
        left, right = x.indptr[i], x.indptr[i + 1]
        for j in range(left, right):
            indices = torch.cat([indices, torch.tensor([[i, x.indices[j]]])])
            values = torch.cat([values, torch.tensor([x.data[j]])])
    return torch.sparse.FloatTensor(indices.t(), values, x.shape)