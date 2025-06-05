import functools
from typing import Any, NamedTuple

import jax


class param_dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def param_dict_flatten(param_dict: param_dict):
    keys, values = zip(*param_dict.items())
    return tuple(zip(map(jax.tree_util.DictKey, keys), values)), keys


jax.tree_util.register_pytree_with_keys(param_dict, param_dict_flatten, lambda *kv: param_dict(zip(*kv)))


class ArrayInfo:
    __slots__ = ("shape", "initializer", "pspec", "dtype")

    def __init__(self, shape, initializer: jax.nn.initializers.Initializer, pspec, dtype=None):
        self.shape = shape
        self.initializer = initializer
        self.pspec = pspec
        self.dtype = dtype

    def __repr__(self):
        return f"ArrayInfo(shape={self.shape}, initializer={repr(self.initializer)}, pspec={self.pspec}, dtype={self.dtype})"

    def init(self, key):
        return self.initializer(key, self.shape, self.dtype)

    def abstract(self, sharding=None):
        return jax.ShapeDtypeStruct(self.shape, self.dtype, sharding=sharding)

    def as_leaf(self, cursor: "InfoTree"):
        self.dtype = self.dtype or cursor._dtype


class InfoTree(NamedTuple):
    dict_ref: param_dict = param_dict()
    path: tuple[str, ...] = ()
    scope_dtype: jax.typing.DTypeLike | None = None

    def asdict(self) -> param_dict:
        return self.dict_ref

    def dtype(self, dtype):
        return InfoTree(self.dict_ref, self.path, dtype)

    def __getattr__(self, key: str):
        return InfoTree(self.dict_ref, self.path + (key,), self.scope_dtype)

    def __getitem__(self, key: Any):  # type: ignore
        return InfoTree(self.dict_ref, self.path + (key,), self.scope_dtype)

    def __setattr__(self, name, value):
        if isinstance(value, ArrayInfo):
            value.as_leaf(self)
        dict.__setitem__(self.parent_ref, name, value)

    def __setitem__(self, name, value):
        if isinstance(value, ArrayInfo):
            value.as_leaf(self)
        dict.__setitem__(self.parent_ref, name, value)

    @property
    def parent_ref(self):
        def descend_default(obj, key):
            if not isinstance(obj, param_dict):
                raise ValueError(f"Unexpected type at key {key} ({type(obj)}), expected a param_dict")
            if key not in obj:
                dict.__setitem__(obj, key, param_dict())
            assert isinstance(new_obj := dict.__getitem__(obj, key), param_dict)
            return new_obj

        return functools.reduce(descend_default, self.path, self.dict_ref)


if __name__ == "__main__":
    import jax.nn.initializers as init

    params = InfoTree()
    params.thing.hello.cat

    def linear(params, inp):
        out = inp
        del inp

        out = params.bias + out @ params.kernel
        return out

    def get_info_tree(in_channels=8, out_channels=16, num_layers=8):
        params = InfoTree()
        for layer in range(num_layers):
            layer_params = params[layer]
            print(layer_params)
            layer_params.bias = ArrayInfo((16,), init.zeros, None, "bfloat16")
            layer_params.kernel = ArrayInfo((8, 16), init.zeros, None, "bfloat16")
        return params.asdict()

    info_tree = get_info_tree()

    print("type", type(info_tree))
    print("keys", info_tree.keys())

    def get_params(info_tree, key):
        treedef = jax.tree.structure(info_tree)
        key_tree = jax.tree.unflatten(treedef, jax.random.split(key, treedef.num_leaves))
        # jax.tree.map(operator.methodcaller("init"), )
        param_tree = jax.tree.map(lambda leaf, key: leaf.init(key), info_tree, key_tree)
        return param_tree

    param_tree = get_params(info_tree, jax.random.key(0))
    print(param_tree)
    inputs = jax.numpy.ones((128, 8))

    def model(params, inp, num_layers=8):
        out = inp
        del inp

        for layer in range(num_layers):
            out = linear(params[layer], out)

        return out
