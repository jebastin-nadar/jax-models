from typing import List

import jax.numpy as jnp


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


def window_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    assert (
        H % window_size[0] == 0
    ), f"height ({H}) must be divisible by window ({window_size[0]})"
    assert (
        W % window_size[1] == 0
    ), f"width ({W}) must be divisible by window ({window_size[1]})"
    x = jnp.reshape(
        x,
        (
            B,
            H // window_size[0],
            window_size[0],
            W // window_size[1],
            window_size[1],
            C,
        ),
    )
    windows = jnp.reshape(
        jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = jnp.reshape(
        windows,
        (
            -1,
            H // window_size[0],
            W // window_size[1],
            window_size[0],
            window_size[1],
            C,
        ),
    )
    x = jnp.reshape(jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, H, W, C))
    return x


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    assert H % grid_size[0] == 0, f"height {H} must be divisible by grid {grid_size[0]}"
    assert W % grid_size[1] == 0, f"width {W} must be divisible by grid {grid_size[1]}"
    x = jnp.reshape(
        x, (B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    )
    windows = jnp.reshape(
        jnp.transpose(x, (0, 2, 4, 1, 3, 5)), (-1, grid_size[0], grid_size[1], C)
    )
    return windows


def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = jnp.reshape(
        windows,
        (-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C),
    )
    x = jnp.reshape(jnp.transpose(x, (0, 3, 1, 4, 2, 5)), (-1, H, W, C))
    return x


def get_rel_pos_ind(window_size: int):
    max_rel_pos = window_size - 1
    n_rel_distance = 2 * max_rel_pos + 1

    rel_pos_ind = jnp.zeros((window_size, window_size, n_rel_distance))
    for i in range(window_size):
        for j in range(window_size):
            k = j - i + max_rel_pos
            if max_rel_pos < abs(j - i):
                continue
            rel_pos_ind = rel_pos_ind.at[i, j, k].set(1)

    return rel_pos_ind


def index_rel_pos_table(rel_pos_table, rel_pos_ind, area):
    reindexed_tensor = jnp.einsum("nhw,ixh->nixw", rel_pos_table, rel_pos_ind)
    reindexed_tensor = jnp.einsum("nixw,jyw->nijxy", reindexed_tensor, rel_pos_ind)

    reindexed_tensor = jnp.reshape(
        reindexed_tensor, (rel_pos_table.shape[0], area, area)
    )
    return reindexed_tensor
