from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
from utils import (
    get_rel_pos_ind,
    grid_partition,
    grid_reverse,
    index_rel_pos_table,
    window_partition,
    window_reverse,
)


class DropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        if self.drop_prob == 0.0 or not train:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape) / keep_prob
        return x * random_tensor


class Stem(nn.Module):
    out_chs: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(
            self.out_chs,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, epsilon=0.001, momentum=0.9, name="norm1"
        )(x)
        # x = nn.Dropout(drop_rate)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Conv(
            self.out_chs, kernel_size=(3, 3), strides=(1, 1), padding=1, name="conv2"
        )(x)
        return x


class SEModule(nn.Module):
    channels: int
    rd_ratio: float = 1.0 / 16

    @nn.compact
    def __call__(self, x, train: bool):
        rd_channels = int(self.channels * self.rd_ratio)

        x_se = jnp.mean(x, axis=(1, 2), keepdims=True)
        x_se = nn.Conv(
            rd_channels, kernel_size=(1, 1), strides=(1, 1), padding=0, name="fc1"
        )(x_se)
        x_se = nn.silu(x_se)
        x_se = nn.Conv(
            self.channels, kernel_size=(1, 1), strides=(1, 1), padding=0, name="fc2"
        )(x_se)
        x_out = x * nn.sigmoid(x_se)
        return x_out


class MbConvBlock(nn.Module):
    inp_chs: int
    out_chs: int
    stride: int = 1
    expand_ratio: int = 4

    @nn.compact
    def __call__(self, x, train: bool):
        mid_chs = self.out_chs * self.expand_ratio

        if self.stride == 2:
            shortcut = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            print(self.inp_chs, self.out_chs)
            if self.inp_chs != self.out_chs:
                shortcut = nn.Conv(
                    self.out_chs,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=0,
                    name="shortcut_expand",
                )(shortcut)
        else:
            shortcut = x
        x = nn.BatchNorm(
            use_running_average=not train, epsilon=0.001, momentum=0.9, name="pre_norm"
        )(x)

        # 1x1 expansion conv & norm-act
        x = nn.Conv(
            mid_chs,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            padding=0,
            name="conv1_1x1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, epsilon=0.001, momentum=0.9, name="norm1"
        )(x)
        x = nn.gelu(x, approximate=True)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = nn.Conv(
            mid_chs,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            use_bias=False,
            padding="SAME",
            feature_group_count=mid_chs,
            name="conv2_kxk",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, epsilon=0.001, momentum=0.9, name="norm2"
        )(x)
        x = nn.gelu(x, approximate=True)

        x = SEModule(mid_chs, name="se")(x, train)

        # 1x1 linear projection to output width
        x = nn.Conv(
            self.out_chs,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            name="conv3_1x1",
        )(x)
        # x = self.drop_path(x) + shortcut
        x_out = x + shortcut
        return x_out


class MLPBlock(nn.Module):
    dim: int
    mlp_dim: int
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.mlp_dim, name="fc1")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.dim, name="fc2")(x)
        return x


class RelPosBias(nn.Module):
    window_size: int
    num_heads: int

    @nn.compact
    def __call__(self, attn):
        size = 2 * self.window_size - 1
        area = self.window_size * self.window_size
        rel_pos_table = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            (self.num_heads, size, size),
        )
        rel_pos_ind = self.variable(
            col="rel_pos_ind",
            name="rel_pos_ind",
            init_fn=lambda: get_rel_pos_ind(self.window_size),
        ).value
        rel_pos_bias = index_rel_pos_table(rel_pos_table, rel_pos_ind, area)
        return attn + rel_pos_bias


class AttentionCl(nn.Module):
    dim: int
    dim_head: int
    window_size: int
    bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        B = x.shape[0]
        restore_shape = x.shape[:-1]
        num_heads = self.dim // self.dim_head
        scale = self.dim_head**-0.5

        qkv = nn.Dense(self.dim * 3, use_bias=self.bias, name="qkv")(x)
        qkv = jnp.swapaxes(jnp.reshape(qkv, (B, -1, 3, num_heads, self.dim_head)), 1, 3)
        q, k, v = [jnp.squeeze(arr, 2) for arr in jnp.split(qkv, 3, axis=2)]

        q = q * scale
        attn = q @ jnp.swapaxes(k, -2, -1)

        attn = RelPosBias(self.window_size, num_heads, name="rel_pos")(attn)

        attn = nn.softmax(attn, axis=-1)
        # attn = self.attn_drop(attn)
        x = attn @ v

        x = jnp.reshape(jnp.swapaxes(x, 1, 2), restore_shape + (-1,))
        x = nn.Dense(self.dim, use_bias=self.bias, name="proj")(x)
        # x = self.proj_drop(x)
        return x


class PartitionAttentionCl(nn.Module):
    dim: int
    dim_head: int
    partition_type: str
    partition_size: List[int]
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.0
    expand_ratio: int = 4
    attn_bias: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        shortcut1 = x

        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(x)
        img_size = x.shape[1:3]
        if self.partition_type == "block":
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = AttentionCl(
            self.dim,
            self.dim_head,
            self.partition_size[0],
            self.attn_bias,
            self.attn_drop,
            self.proj_drop,
            name="attn",
        )(partitioned, train)

        if self.partition_type == "block":
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse(partitioned, self.partition_size, img_size)

        # x = drop_path1(x)

        x = shortcut1 + x
        shortcut2 = x

        x = nn.LayerNorm(epsilon=1e-5, name="norm2")(x)
        x = MLPBlock(self.dim, int(self.dim * self.expand_ratio), name="mlp")(x, train)
        # x = drop_path2(x)

        x = shortcut2 + x
        return x
