import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
class MLP(nn.Module):
    def __init__(self, in_dim:int, layers: List[int],
                 activation: str = 'relu',
                 dropout: float = 0,
                 batch_norm: bool = True):
        # 可学习的参数要在init中注册
        super(MLP, self).__init__()
        self.mlp_layers = nn.ModuleList()
        for i, out_dim in enumerate(layers):
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                self.mlp_layers.append(F.ReLU())
            elif activation == "sigmoid":
                self.mlp_layers.append(F.Sigmoid())
            elif activation == "tanh":
                self.mlp_layers.append(F.Tanh())
            else:
                raise ValueError("Activation function not supported.")
            if dropout > 0:
                self.mlp_layers.append(nn.Dropout(dropout))
            if batch_norm and i != len(layers) - 1:
                self.mlp_layers.append(nn.BatchNorm1d(out_dim)) # nn.batchnorm1d (B,N)
            in_dim = out_dim
    # activation 已经加在layer里了，不需要单独调用
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            x = layer(x)
        return x
    

# test MLP module
# 
model= MLP(1024,[256,64,2])
print(model.state_dict())
print(model.__module__)