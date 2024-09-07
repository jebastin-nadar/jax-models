from typing import List

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from layers import MbConvBlock, PartitionAttentionCl, Stem


class MaxVitBlock(nn.Module):
    inp_dim: int
    dim: int
    dim_head: int
    partition_size: List[int]
    stride: int = 1
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        x = MbConvBlock(self.inp_dim, self.dim, self.stride, name="conv")(x, train)
        x = PartitionAttentionCl(
            self.dim,
            self.dim_head,
            "block",
            self.partition_size,
            self.attn_drop,
            self.proj_drop,
            self.drop_path,
            name="attn_block",
        )(x, train)
        x = PartitionAttentionCl(
            self.dim,
            self.dim_head,
            "grid",
            self.partition_size,
            self.attn_drop,
            self.proj_drop,
            self.drop_path,
            name="attn_grid",
        )(x, train)
        return dict(x=x, train=train)


class MaxVitStage(nn.Module):
    inp_dim: int
    dim: int
    dim_head: int
    num_blocks: int
    partition_size: List[int]
    drop_path: List[float]
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        blocks = []
        for i in range(self.num_blocks):
            block_stride = 2 if i == 0 else 1
            blocks.append(
                MaxVitBlock(
                    self.inp_dim,
                    self.dim,
                    self.dim_head,
                    self.partition_size,
                    block_stride,
                    self.attn_drop,
                    self.proj_drop,
                    self.drop_path[i],
                    name=f"blocks_{i}",
                )
            )

        x = nn.Sequential(blocks)(x, train)["x"]
        return dict(x=x, train=train)


class MaxVit(nn.Module):
    img_size: List[int]
    stem_width: int
    emb_dims: List[int]
    depths: List[int]
    dim_head: int = 32
    num_classes: int = 1000
    features_only: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        x = Stem(self.stem_width, name="stem")(x, train)

        dpr = np.split(
            np.linspace(0, self.drop_path_rate, sum(self.depths)),
            np.cumsum(self.depths)[:-1],
        )
        partition_size = [self.img_size[0] // 32, self.img_size[1] // 32]
        inp_dim = self.stem_width

        feat_maps = []
        for i, depth in enumerate(self.depths):
            x = MaxVitStage(
                inp_dim,
                self.emb_dims[i],
                self.dim_head,
                depth,
                partition_size,
                dpr[i],
                self.attn_drop,
                self.proj_drop,
                name=f"stages_{i}",
            )(x, train)["x"]
            feat_maps.append(x)
            inp_dim = self.emb_dims[i]

        if self.features_only:
            return feat_maps
        else:
            x = jnp.mean(x, axis=(1, 2))
            x = nn.LayerNorm(epsilon=1e-5, name="head_norm")(x)
            x = nn.Dense(self.emb_dims[-1], name="head_pre_logits_fc")(x)
            x = nn.tanh(x)
            x = nn.Dropout(self.drop_rate)(x)
            logits = nn.Dense(self.num_classes, name="head_fc")(x)
            return logits
