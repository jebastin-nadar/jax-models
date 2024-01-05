from typing import List

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from layers import ConvBnAct, EdgeResidual, InvertedResidual


class EfficientNetV2(nn.Module):
    stem_width: int
    emb_dims: List[int]
    depths: List[int]
    block_types: List[str]
    strides: List[int]
    expand_ratios: List[float]
    rd_ratios: List[float]
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    features_only: bool = False
    num_classes: int = 1000
    head_width: int = 1024

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(
            self.stem_width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1),
            use_bias=False,
            name="conv_stem",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn1")(x)
        x = nn.silu(x)

        dpr = np.split(
            np.round(np.linspace(0, self.drop_path_rate, sum(self.depths)), 3),
            np.cumsum(self.depths)[:-1],
        )
        feat_maps = []

        for i in range(len(self.emb_dims)):
            block_type = self.block_types[i]
            stride = self.strides[i]
            exp_ratio = self.expand_ratios[i]
            rd_ratio = self.rd_ratios[i]
            emb_dim = self.emb_dims[i]

            stages = []
            if i == 0:
                inp_dim = self.stem_width
            else:
                inp_dim = self.emb_dims[i - 1]

            for j in range(self.depths[i]):
                if block_type == "cn":
                    blk = ConvBnAct(
                        inp_dim, emb_dim, stride, dpr[i][j], name=f"blocks_{i}_{j}"
                    )
                elif block_type == "er":
                    blk = EdgeResidual(
                        inp_dim,
                        emb_dim,
                        stride,
                        exp_ratio,
                        dpr[i][j],
                        name=f"blocks_{i}_{j}",
                    )
                elif block_type == "ir":
                    blk = InvertedResidual(
                        inp_dim,
                        emb_dim,
                        stride,
                        exp_ratio,
                        rd_ratio,
                        dpr[i][j],
                        name=f"blocks_{i}_{j}",
                    )
                stages.append(blk)

                inp_dim = emb_dim
                stride = 1

            x = nn.Sequential(stages)(x, train)["x"]
            feat_maps.append(x)

        if self.features_only:
            return feat_maps
        else:
            x = nn.Conv(
                self.head_width,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                use_bias=False,
                name="conv_head",
            )(feat_maps[-1])
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn2")(x)
            x = nn.silu(x)
            x = jnp.mean(x, axis=(1, 2))
            # x = nn.Dropout(self.drop_rate)(x)
            logits = nn.Dense(self.num_classes, name="classifier")(x)
            return logits
