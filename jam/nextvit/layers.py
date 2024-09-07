import flax.linen as nn
import jax
import jax.numpy as jnp


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        if self.drop_prob == 0.0 or not train:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng("dropout")
        mask = jax.random.bernoulli(key=rng, p=keep_prob, shape=shape)
        mask = jnp.broadcast_to(mask, x.shape)
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class ConvBNReLU(nn.Module):
    out_channels: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=(1, 1),
            use_bias=False,
            name="conv",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(x)
        x = nn.relu(x)
        return dict(x=x, train=train)


class PatchEmbed(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        if self.stride == 2:
            x = nn.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), count_include_pad=False
            )
            x = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False,
                name="conv",
            )(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(
                x
            )
        elif self.in_channels != self.out_channels:
            x = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False,
                name="conv",
            )(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(
                x
            )
        else:
            pass
        return x


class Mlp(nn.Module):
    in_features: int
    mlp_ratio: float = None
    drop: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        hidden_dim = _make_divisible(self.in_features * self.mlp_ratio, 32)
        x = nn.Conv(
            hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=self.bias,
            name="conv1",
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(self.drop, deterministic=not train, name="drop1")(x)
        x = nn.Conv(
            self.in_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=self.bias,
            name="conv2",
        )(x)
        x = nn.Dropout(self.drop, deterministic=not train, name="drop2")(x)
        return x


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    out_channels: int
    head_dim: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            feature_group_count=self.out_channels // self.head_dim,
            use_bias=False,
            name="group_conv3x3",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            use_bias=False,
            name="projection",
        )(x)
        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """

    in_channels: int
    out_channels: int
    stride: int = 1
    path_dropout: float = 0.0
    drop: float = 0.0
    head_dim: int = 32
    mlp_ratio: float = 3.0

    @nn.compact
    def __call__(self, x, train: bool):
        x = PatchEmbed(
            self.in_channels, self.out_channels, self.stride, name="patch_embed"
        )(x, train)
        x1 = MHCA(self.out_channels, self.head_dim, name="mhca")(x, train)
        x1 = DropPath(self.path_dropout, name="attention_path_dropout")(x1, train)
        x = x + x1

        out = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(x)
        out = Mlp(self.out_channels, self.mlp_ratio, self.drop, bias=True, name="mlp")(
            out, train
        )
        out = DropPath(self.path_dropout, name="mlp_path_dropout")(out, train)
        x = x + out
        return dict(x=x, train=train)


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """

    dim: int
    head_dim: int = 32
    sr_ratio: float = 1.0
    qkv_bias: bool = True
    qk_scale: float = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        num_heads = self.dim // self.head_dim
        scale = self.qk_scale or self.head_dim**-0.5

        B, N, C = x.shape
        q = nn.Dense(self.dim, use_bias=self.qkv_bias, name="q")(x)
        q = jnp.swapaxes(jnp.reshape(q, (B, N, num_heads, C // num_heads)), 1, 2)

        if self.sr_ratio > 1:
            N_ratio = self.sr_ratio**2
            x = nn.avg_pool(x, window_shape=(N_ratio,), strides=(N_ratio,))
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm")(
                x
            )
            k = nn.Dense(self.dim, use_bias=self.qkv_bias, name="k")(x)
            k = jnp.transpose(
                jnp.reshape(k, (B, -1, num_heads, C // num_heads)), (0, 2, 3, 1)
            )
            v = nn.Dense(self.dim, use_bias=self.qkv_bias, name="v")(x)
            v = jnp.swapaxes(jnp.reshape(v, (B, -1, num_heads, C // num_heads)), 1, 2)
        else:
            k = nn.Dense(self.dim, use_bias=self.qkv_bias, name="k")(x)
            k = jnp.transpose(
                jnp.reshape(k, (B, -1, num_heads, C // num_heads)), (0, 2, 3, 1)
            )
            v = nn.Dense(self.dim, use_bias=self.qkv_bias, name="v")(x)
            v = jnp.swapaxes(jnp.reshape(v, (B, -1, num_heads, C // num_heads)), 1, 2)

        attn = (q @ k) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop, deterministic=not train, name="attn_drop")(
            attn
        )

        x = jnp.reshape(jnp.swapaxes(attn @ v, 1, 2), (B, N, C))
        x = nn.Dense(self.dim, name="proj")(x)
        x = nn.Dropout(self.proj_drop, deterministic=not train, name="proj_drop")(x)
        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """

    in_channels: int
    out_channels: int
    path_dropout: float = 0.0
    stride: int = 1
    sr_ratio: float = 1
    mlp_ratio: float = 2
    head_dim: int = 32
    mix_block_ratio: float = 0.75
    attn_drop: float = 0.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        mhsa_out_channels = _make_divisible(
            int(self.out_channels * self.mix_block_ratio), 32
        )
        mhca_out_channels = self.out_channels - mhsa_out_channels

        x = PatchEmbed(
            self.in_channels, mhsa_out_channels, self.stride, name="patch_embed"
        )(x, train)
        B, H, W, C = x.shape
        out = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm1")(x)
        out = jnp.reshape(out, (B, H * W, C))
        out = E_MHSA(
            mhsa_out_channels,
            self.head_dim,
            self.sr_ratio,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            name="e_mhsa",
        )(out, train)
        out = DropPath(
            self.path_dropout * self.mix_block_ratio, name="mhsa_path_dropout"
        )(out, train)
        x = x + jnp.reshape(out, (B, H, W, C))

        out = PatchEmbed(
            mhsa_out_channels, mhca_out_channels, stride=1, name="projection"
        )(x, train)
        out2 = MHCA(mhca_out_channels, head_dim=self.head_dim, name="mhca")(out, train)
        out2 = DropPath(
            self.path_dropout * (1 - self.mix_block_ratio), name="mhca_path_dropout"
        )(out2, train)
        out = out + out2

        x = jnp.concatenate([x, out], axis=-1)

        out = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="norm2")(x)
        out = Mlp(self.out_channels, self.mlp_ratio, self.drop, name="mlp")(out, train)
        out = DropPath(self.path_dropout, name="mlp_path_dropout")(out, train)
        x = x + out
        return dict(x=x, train=train)
