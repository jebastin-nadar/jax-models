import flax.linen as nn
import jax
import jax.numpy as jnp


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


class SqueezeExcite(nn.Module):
    channels: int
    rd_ratio: float = 1.0 / 16

    @nn.compact
    def __call__(self, x, train: bool):
        rd_channels = int(self.channels * self.rd_ratio)

        x_se = jnp.mean(x, axis=(1, 2), keepdims=True)
        x_se = nn.Conv(
            rd_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            name="conv_reduce",
        )(x_se)
        x_se = nn.silu(x_se)
        x_se = nn.Conv(
            self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            name="conv_expand",
        )(x_se)
        x_out = x * nn.sigmoid(x_se)
        return x_out


class ConvBnAct(nn.Module):
    in_chs: int
    out_chs: int
    stride: int = 1
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        has_skip = self.stride == 1 and self.in_chs == self.out_chs

        shortcut = x
        x = nn.Conv(
            self.out_chs,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            use_bias=False,
            name="conv",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn1")(x)
        x = nn.silu(x)
        if has_skip:
            x = DropPath(self.drop_path_rate, name="drop_path")(x, train) + shortcut
        return dict(x=x, train=train)


class EdgeResidual(nn.Module):
    in_chs: int
    out_chs: int
    stride: int = 1
    exp_ratio: float = 1.0
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        mid_chs = int(self.in_chs * self.exp_ratio)
        has_skip = self.in_chs == self.out_chs and self.stride == 1

        shortcut = x
        x = nn.Conv(
            mid_chs,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=(1, 1),
            use_bias=False,
            name="conv_exp",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn1")(x)
        x = nn.silu(x)
        x = nn.Conv(
            self.out_chs,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=False,
            name="conv_pwl",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn2")(x)

        if has_skip:
            x = DropPath(self.drop_path_rate, name="drop_path")(x, train) + shortcut
        return dict(x=x, train=train)


class InvertedResidual(nn.Module):
    in_chs: int
    out_chs: int
    stride: int = 1
    exp_ratio: float = 1.0
    rd_ratio: float = 1.0 / 24
    drop_path_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        mid_chs = int(self.in_chs * self.exp_ratio)
        has_skip = self.in_chs == self.out_chs and self.stride == 1

        shortcut = x
        x = nn.Conv(
            mid_chs,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=False,
            name="conv_pw",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn1")(x)
        x = nn.silu(x)
        x = nn.Conv(
            mid_chs,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=(1, 1),
            use_bias=False,
            feature_group_count=mid_chs,
            name="conv_dw",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn2")(x)
        x = nn.silu(x)
        x = SqueezeExcite(mid_chs, self.rd_ratio, name="se")(x, train)
        x = nn.Conv(
            self.out_chs,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=False,
            name="conv_pwl",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, name="bn3")(x)

        if has_skip:
            x = DropPath(self.drop_path_rate, name="drop_path")(x, train) + shortcut

        return dict(x=x, train=train)
