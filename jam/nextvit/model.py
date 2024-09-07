from typing import List

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from layers import ConvBNReLU, NCB, NTB


class NextViT(nn.Module):
    stem_chs: List[int]
    depths: List[int]
    num_classes: int = 1000
    features_only: bool = False
    drop_path_rate: float = 0.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    strides: List[int] = (2, 2, 2, 2)
    sr_ratios: List[int] = (8, 4, 2, 1)
    head_dim: int = 32
    mix_block_ratio: float = 0.75

    @nn.compact
    def __call__(self, x, train: bool):
        stage_out_channels = [
            [96] * (self.depths[0]),
            [192] * (self.depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (self.depths[2] // 5),
            [768] * (self.depths[3] - 1) + [1024],
        ]

        stage_block_types = [
            ["NCB"] * self.depths[0],
            ["NCB"] * (self.depths[1] - 1) + ["NTB"],
            ["NCB", "NCB", "NCB", "NCB", "NTB"] * (self.depths[2] // 5),
            ["NCB"] * (self.depths[3] - 1) + ["NTB"],
        ]

        feat_maps = []

        x = nn.Sequential(
            [
                ConvBNReLU(self.stem_chs[0], 3, 2, name="stem_0"),
                ConvBNReLU(self.stem_chs[1], 3, 1, name="stem_1"),
                ConvBNReLU(self.stem_chs[2], 3, 1, name="stem_2"),
                ConvBNReLU(self.stem_chs[2], 3, 1, name="stem_3"),
            ]
        )(x, train)["x"]
        feat_maps.append(x)

        input_channel = self.stem_chs[-1]
        idx = 0
        dpr = np.linspace(0, self.drop_path_rate, sum(self.depths))
        for stage_id in range(len(self.depths)):
            numrepeat = self.depths[stage_id]
            output_channels = stage_out_channels[stage_id]
            block_types = stage_block_types[stage_id]
            blocks = []
            for block_id in range(numrepeat):
                if self.strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type == "NCB":
                    layer = NCB(
                        input_channel,
                        output_channel,
                        stride,
                        dpr[idx + block_id],
                        self.proj_drop,
                        self.head_dim,
                        name=f"features_{idx + block_id}",
                    )
                    blocks.append(layer)
                elif block_type == "NTB":
                    layer = NTB(
                        input_channel,
                        output_channel,
                        dpr[idx + block_id],
                        stride,
                        self.sr_ratios[stage_id],
                        head_dim=self.head_dim,
                        mix_block_ratio=self.mix_block_ratio,
                        attn_drop=self.attn_drop,
                        drop=self.proj_drop,
                        name=f"features_{idx + block_id}",
                    )
                    blocks.append(layer)
                input_channel = output_channel
            idx += numrepeat
            x = nn.Sequential(blocks)(x, train)["x"]
            feat_maps.append(x)

        if self.features_only:
            return feat_maps
        else:
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(
                x
            )
            x = jnp.mean(x, axis=(1, 2))
            x = nn.Dropout(self.drop_rate)(x)
            logits = nn.Dense(self.num_classes, name="head_fc")(x)
            return logits
