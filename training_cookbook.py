import functools as ft
import itertools as it
import operator as op
import time
from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import mutable_array
from jax.sharding import PartitionSpec

ode = (
    "We are the music makers,\n"
    "    And we are the dreamers of dreams,\n"
    "Wandering by lone sea-breakers,\n"
    "    And sitting by desolate streams;-\n"
    "World-losers and world-forsakers,\n"
    "    On whom the pale moon gleams:\n"
    "Yet we are the movers and shakers\n"
    "    Of the world for ever, it seems.\n"
)


@jax.tree_util.register_static
@dataclass(kw_only=True)
class Config:
    mesh_axis_names: tuple[str, ...] = ("fsdp",)
    mesh_shape: tuple[int, ...] = (8,)
    seq_length: int = 128

    num_train_steps: int = 10**6
    host_batch_size: int = 16
    learning_rate: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0

    param_seed: int = 12738
    num_layers: int = 4
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    vocab_size: int = 2**8  # uint8 ascii encoding
    num_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"

    # tag: sharding
    embed: PartitionSpec = PartitionSpec(None, None)
    pos_embed: PartitionSpec = PartitionSpec(None, None)
    att_qkv: PartitionSpec = PartitionSpec(None, "fsdp", None, None)
    att_out: PartitionSpec = PartitionSpec("fsdp", None, None)
    mlp_in: PartitionSpec = PartitionSpec("fsdp", None)
    mlp_out: PartitionSpec = PartitionSpec(None, "fsdp")
    in_kernel: PartitionSpec = PartitionSpec(None, None)
    in_bias: PartitionSpec = PartitionSpec(None)
    out_kernel: PartitionSpec = PartitionSpec("fsdp", None)
    out_bias: PartitionSpec = PartitionSpec(None)

    act_ids: PartitionSpec = PartitionSpec("fsdp")
    act_seq: PartitionSpec = PartitionSpec("fsdp", None, None)
    act_att: PartitionSpec = PartitionSpec("fsdp", None, None, None)
    act_hidden: PartitionSpec = PartitionSpec("fsdp", None, None)
    # tag: sharding


@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


# tag: get-param-state
def get_param_state(config: Config) -> dot_dict:
    root_key = jax.random.key(config.param_seed)
    key = map(ft.partial(jax.random.fold_in, root_key), it.count())
    zero_init = jax.nn.initializers.constant(0.0)
    he_init = jax.nn.initializers.he_normal(1, 1)
    dtype = config.dtype

    params = dot_dict(
        pos_embed=zero_init(next(key), (config.seq_length, config.embed_dim), dtype, config.pos_embed),
        layers=dot_dict(),
    )
    params.embedding = he_init(next(key), (config.vocab_size, config.embed_dim), dtype, config.embed)
    params.linear_in = dot_dict(
        kernel=he_init(next(key), (1, config.embed_dim), dtype, config.in_kernel),
        bias=zero_init(next(key), (config.embed_dim,), dtype, config.in_bias),
    )
    params.linear_out = dot_dict(
        kernel=he_init(next(key), (config.embed_dim, config.vocab_size), dtype, config.out_kernel),
    )
    for layer in range(config.num_layers):
        qkv_shape = (3, config.embed_dim, config.num_heads, config.head_dim)
        out_shape = (config.num_heads, config.head_dim, config.embed_dim)
        params.layers[layer] = dot_dict(
            attention=dot_dict(
                qkv=he_init(next(key), qkv_shape, dtype, config.att_qkv),
                out=he_init(next(key), out_shape, dtype, config.att_out),
            ),
            mlp=dot_dict(
                in_kernel=he_init(next(key), (config.embed_dim, config.mlp_dim), dtype, config.mlp_in),
                out_kernel=he_init(next(key), (config.mlp_dim, config.embed_dim), dtype, config.mlp_out),
            ),
        )
    return params  # tag: get-param-state


# tag: model-apply
def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
    out = tokens
    del tokens
    lin_einsum = ft.partial(jnp.einsum, out_sharding=config.act_seq)

    out =
    out = lin_einsum("bs,sd->bsd", out, params.linear_in.kernel) + params.linear_in.bias
    out += params.pos_embed

    for layer in range(config.num_layers):
        block = params.layers[layer]
        att_skip = out  # 1 billion dollars in venture capital funding please
        qkv = jnp.einsum("bsd,3dkh->bs3kh", out, block.attention.qkv, out_sharding=config.act_att)
        out = jax.nn.dot_product_attention(qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :], is_causal=True)
        out = lin_einsum("bskh,khd->bsd", out, block.attention.out)
        out += att_skip
        out *= jax.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

        mlp_skip = out  # machine learning circa 1986
        out = jnp.einsum("bsd,dh->bsh", out, block.mlp.in_kernel, out_sharding=config.act_hidden)
        out = jax.nn.gelu(out)
        out = lin_einsum("bsh,hd->bsd", out, block.mlp.out_kernel)
        out += mlp_skip
        out *= jax.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

    logits = lin_einsum("bsd,dl->bsl", out, params.linear_out.kernel)
    return logits  # tag: model-apply


# tag: get-adam-state
def get_adam_state(param: jax.Array) -> dot_dict:
    adam_state = dot_dict(mu=jnp.zeros_like(param), nu=jnp.zeros_like(param), count=jnp.array(0))
    return adam_state  # tag: get-adam-state


# tag: adam-apply
def adam_apply(config: Config, param: jax.Array, grad: jax.Array, adam_state: dot_dict):
    def update_moment(grad, moment, decay, order):
        return (1 - decay) * (grad**order) + decay * moment

    def bias_correction(moment, decay, count):
        return moment / (1 - decay**count).astype(moment.dtype)

    adam_state.mu[...] = update_moment(grad, adam_state.mu[...], config.beta_1, 1)
    adam_state.nu[...] = update_moment(grad, adam_state.nu[...], config.beta_2, 2)
    adam_state.count[...] = adam_state.count[...] + 1

    mu_hat = bias_correction(adam_state.mu[...], config.beta_1, adam_state.count[...])
    nu_hat = bias_correction(adam_state.nu[...], config.beta_2, adam_state.count[...])
    update = mu_hat / (jnp.sqrt(nu_hat + config.eps_root) + config.eps)
    param[...] = param[...] - config.learning_rate * update  # tag: adam-apply


# tag: get-train-state
@jax.jit
def get_train_state(config: Config) -> dot_dict:
    train_state = dot_dict()
    train_state.params = get_param_state(config)
    train_state.opt = jax.tree.map(get_adam_state, train_state.params)
    return train_state  # tag: get-train-state


# tag: train-step
@jax.jit
def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
    def loss_fn(params):
        logits = model_apply(config, params, batch["observed_ids"])
        labels = jax.nn.one_hot(batch["target_ids"], config.vocab_size)
        return -(labels * jax.nn.log_softmax(logits)).mean()

    params = jax.tree.map(op.methodcaller("__getitem__", slice(None, None, None)), train_state.params)
    loss, grad = jax.value_and_grad(loss_fn)(params)
    jax.tree.map(ft.partial(adam_apply, config), train_state.params, grad, train_state.opt)
    metrics = {"train_loss": loss}
    return metrics  # tag: train-step


# tag: record-writer
class RecordWriter:
    prev_metrics = None

    def __call__(self, cur_metrics: dict):
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
        if log_metrics is None:
            return
        print(*it.starmap("{}: {}".format, log_metrics.items()), sep="\t")
        # tag: record-writer


# tag: get-dataset
def get_dataset(config: Config, single_batch=ode) -> Iterator[dict[str, np.ndarray]]:
    while True:
        observed_array = np.frombuffer(single_batch.encode("ascii"), dtype=np.uint8)
        target_array = np.roll(observed_array, -1)
        time.sleep(0.5)
        yield {  # repeat the sequence across the batch size to simulate multiple data points
            "observed_ids": np.tile(observed_array[: config.seq_length], (config.host_batch_size, 1)),
            "target_ids": np.tile(target_array[: config.seq_length], (config.host_batch_size, 1)),
        }
        # tag: get-dataset


# tag: get-dataset-on-device
def get_dataset_on_device(config: Config) -> Iterator[dict[str, jax.Array]]:
    datset = get_dataset(config)
    sharding = PartitionSpec(config.mesh_axis_names)
    return map(ft.partial(jax.make_array_from_process_local_data, sharding), datset)  # type: ignore
    # tag: get-dataset-on-device


# tag: train-loop
def train_loop(config: Config):
    mesh = jax.make_mesh(
        config.mesh_shape, config.mesh_axis_names, axis_types=(jax.sharding.AxisType.Explicit,)
    )
    jax.sharding.set_mesh(mesh)

    record_writer = RecordWriter()
    train_state = get_train_state(config)
    train_state = jax.tree.map(mutable_array, train_state)
    batch = iter(get_dataset_on_device(config))
    for step in range(config.num_train_steps):
        metrics = train_step(config, train_state, next(batch))
        record_writer({"step": step} | metrics)


# tag: train-loop

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_num_cpu_devices", 8)
    train_loop(config=Config())
