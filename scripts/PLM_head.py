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
            if i == len(layers) - 1: # last layer, no activation function and dropout
                break
            if activation == "relu":
                self.mlp_layers.append(nn.ReLU())
            elif activation == "sigmoid":
                self.mlp_layers.append(nn.Sigmoid())
            elif activation == "tanh":
                self.mlp_layers.append(nn.Tanh())
            elif activation == "gelu":
                self.mlp_layers.append(nn.GELU())
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


class SpeciesAdapter(nn.Module):
    """
    Non-linear adapter that maps a pre-computed species embedding
    (e.g. 768-d PubMedBERT vector) to a compact representation that
    can be concatenated with the protein-LM output.

        in_dim  ──Linear──►  bottleneck  ──LayerNorm──► GELU ──Dropout──►
                Linear ──► out_dim ──► GELU

    Both hidden activations are non-linear (GELU). 
    """
    def __init__(self,
                 in_dim: int = 768,
                 bottleneck: int = 128,
                 out_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> [B, out_dim]
        return self.net(x)
    

# test MLP module
# model= MLP(10,[5,3,2])
# for name, param in model.named_parameters():
#     print(name, param.size())
# x = torch.randn(4,10)
# out = model(x)
# print(out.size())  # should be [4,2]