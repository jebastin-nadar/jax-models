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
    num_classes: int
    stem_width: int
    emb_dims: List[int]
    depths: List[int]
    dim_head: int = 32
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

        stages = []
        for i, depth in enumerate(self.depths):
            stages.append(
                MaxVitStage(
                    inp_dim,
                    self.emb_dims[i],
                    self.dim_head,
                    depth,
                    partition_size,
                    dpr[i],
                    self.drop_rate,
                    self.drop_rate,
                    name=f"stages_{i}",
                )
            )
            inp_dim = self.emb_dims[i]

        x = nn.Sequential(stages)(x, train)["x"]
        x = jnp.mean(x, axis=(1, 2))
        x = nn.LayerNorm(epsilon=1e-5, name="head_norm")(x)
        x = nn.Dense(self.emb_dims[-1], name="head_pre_logits_fc")(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_classes, name="head_fc")(x)
        return x
