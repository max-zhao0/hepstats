import optree
from optree.typing import PyTree
from optree.integrations.numpy import tree_ravel
from collections.abc import Callable, Sequence
from typing import Any, Iterable
import numpy as np

_Node = Any
sentinel = object()

class _LeafWrapper:
    def __init__(self, value: Any):
        self.value = value

def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    if not isinstance(x, _LeafWrapper):
        raise TypeError(f"Operation undefined, {x} is not a leaf of the pytree.")
    return x.value

class _CountedIdDict:
    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self._dict = {id(k): v for k, v in zip(keys, values)}
        self._count = {id(k): 0 for k in keys}

    def __contains__(self, item):
        return id(item) in self._dict

    def __getitem__(self, item):
        self._count[id(item)] += 1
        return self._dict[id(item)]

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def count(self, item):
        return self._count[id(item)]

class _DistinctTuple(tuple):
    pass

def tree_at(
    where: Callable[[PyTree], _Node | Sequence[_Node]],
    pytree: PyTree,
    replace: Any | Sequence[Any] = sentinel,
    replace_fn: Callable[[_Node], Any] = sentinel,
    is_leaf: Callable[[Any], bool] | None = None,
):
    """
    NOTE: Code copied from equinox: https://github.com/patrick-kidger/equinox/blob/f8ca3458d85c178a2addaff7c50ef6f2eb250ced/equinox/_tree.py#L58
    
    Modifies a leaf or subtree of a PyTree. (A bit like using `.at[].set()` on a JAX
    array.)

    The modified PyTree is returned and the original input is left unchanged. Make sure
    to use the return value from this function!

    **Arguments:**

    - `where`: A callable `PyTree -> Node` or `PyTree -> tuple[Node, ...]`. It should
        consume a PyTree with the same structure as `pytree`, and return the node or
        nodes that should be replaced. For example
        `where = lambda mlp: mlp.layers[-1].linear.weight`.
    - `pytree`: The PyTree to modify.
    - `replace`: Either a single element, or a sequence of the same length as returned
        by `where`. This specifies the replacements to make at the locations specified
        by `where`. Mutually exclusive with `replace_fn`.
    - `replace_fn`: A function `Node -> Any`. It will be called on every node specified
        by `where`. The return value from `replace_fn` will be used in its place.
        Mutually exclusive with `replace`.
    - `is_leaf`: As `jtu.tree_flatten`. For example pass `is_leaf=lambda x: x is None`
        to be able to replace `None` values using `tree_at`.

    Note that `where` should not depend on the type of any of the leaves of the
    pytree, e.g. given `pytree = [1, 2, object(), 3]`, then
    `where = lambda x: tuple(xi for xi in x if type(xi) is int)` is not allowed. If you
    really need this behaviour then this example could instead be expressed as
    `where = lambda x: tuple(xi for xi, yi in zip(x, pytree) if type(yi) is int)`.

    **Returns:**

    A copy of the input PyTree, with the appropriate modifications.

    !!! Example

        ```python
        # Here is a pytree
        tree = [1, [2, {"a": 3, "b": 4}]]
        new_leaf = 5
        get_leaf = lambda t: t[1][1]["a"]
        new_tree = eqx.tree_at(get_leaf, tree, 5)
        # new_tree is [1, [2, {"a": 5, "b": 4}]]
        # The original tree is unchanged.
        ```

    !!! Example

        This is useful for performing model surgery. For example:
        ```python
        mlp = eqx.nn.MLP(...)
        new_linear = eqx.nn.Linear(...)
        get_last_layer = lambda m: m.layers[-1]
        new_mlp = eqx.tree_at(get_last_layer, mlp, new_linear)
        ```
        See also the [Tricks](../tricks.md) page.

    !!! Info

        Constructing analogous PyTrees, with the same structure but different leaves, is
        very common in JAX: for example when constructing the `in_axes` argument to
        `jax.vmap`.

        To support this use-case, the returned PyTree is constructed without calling
        `__init__`, `__post_init__`, or
        [`__check_init__`](./module/advanced_fields.md#checking-invariants). This allows
        for modifying leaves to be anything, regardless of the use of any custom
        constructor or custom checks in the original PyTree.
    """  # noqa: E501

    # We need to specify a particular node in a PyTree.
    # This is surprisingly difficult to do! As far as I can see, pretty much the only
    # way of doing this is to specify e.g. `x.foo[0].bar` via `is`, and then pulling
    # a few tricks to try and ensure that the same object doesn't appear multiple
    # times in the same PyTree.
    #
    # So this first `tree_map` serves a dual purpose.
    # 1) Makes a copy of the composite nodes in the PyTree, to avoid aliasing via
    #    e.g. `pytree=[(1,)] * 5`. This has the tuple `(1,)` appear multiple times.
    # 2) It makes each leaf be a unique Python object, as it's wrapped in
    #    `_LeafWrapper`. This is needed because Python caches a few builtin objects:
    #    `assert 0 + 1 is 1`. I think only a few leaf types are subject to this.
    # So point 1) should ensure that all composite nodes are unique Python objects,
    # and point 2) should ensure that all leaves are unique Python objects.
    # Between them, all nodes of `pytree` are handled.
    #
    # I think pretty much the only way this can fail is when using a custom node with
    # singleton-like flatten+unflatten behaviour, which is pretty edge case. And we've
    # added a check for it at the bottom of this function, just to be sure.
    #
    # Whilst we're here: we also double-check that `where` is well-formed and doesn't
    # use leaf information. (As else `node_or_nodes` will be wrong.)
    is_empty_tuple = (
        lambda x: isinstance(x, tuple) and not hasattr(x, "_fields") and x == ()
    )
    pytree = optree.tree_map(
        lambda x: _DistinctTuple() if is_empty_tuple(x) else x,
        pytree,
        is_leaf=is_empty_tuple,
    )
    node_or_nodes_nowrapper = where(pytree)
    pytree = optree.tree_map(_LeafWrapper, pytree, is_leaf=is_leaf)
    node_or_nodes = where(pytree)
    leaves1, structure1 = optree.tree_flatten(node_or_nodes_nowrapper, is_leaf=is_leaf)
    leaves2, structure2 = optree.tree_flatten(node_or_nodes)
    leaves2 = [_remove_leaf_wrapper(x) for x in leaves2]
    if (
        structure1 != structure2
        or len(leaves1) != len(leaves2)
        or any(l1 is not l2 for l1, l2 in zip(leaves1, leaves2))
    ):
        raise ValueError(
            "`where` must use just the PyTree structure of `pytree`. `where` must not "
            "depend on the leaves in `pytree`."
        )
    del node_or_nodes_nowrapper, leaves1, structure1, leaves2, structure2

    # Normalise whether we were passed a single node or a sequence of nodes.
    in_pytree = False

    def _in_pytree(x):
        nonlocal in_pytree
        if x is node_or_nodes:  # noqa: F821
            in_pytree = True
        return x  # needed for jax.tree_util.Partial, which has a dodgy constructor

    optree.tree_map(_in_pytree, pytree, is_leaf=lambda x: x is node_or_nodes)  # noqa: F821
    if in_pytree:
        nodes = (node_or_nodes,)
        if replace is not sentinel:
            replace = (replace,)
    else:
        nodes = node_or_nodes
    del in_pytree, node_or_nodes

    # Normalise replace vs replace_fn
    if replace is sentinel:
        if replace_fn is sentinel:
            raise ValueError(
                "Precisely one of `replace` and `replace_fn` must be specified."
            )
        else:

            def _replace_fn(x):
                x = optree.tree_map(_remove_leaf_wrapper, x)
                return replace_fn(x)

            replace_fns = [_replace_fn] * len(nodes)
    else:
        if replace_fn is sentinel:
            if len(nodes) != len(replace):
                raise ValueError(
                    "`where` must return a sequence of leaves of the same length as "
                    "`replace`."
                )
            replace_fns = [lambda _, r=r: r for r in replace]
        else:
            raise ValueError(
                "Precisely one of `replace` and `replace_fn` must be specified."
            )
    node_replace_fns = _CountedIdDict(nodes, replace_fns)

    # Actually do the replacement
    def _make_replacement(x: _Node) -> Any:
        return node_replace_fns.get(x, _remove_leaf_wrapper)(x)

    out = optree.tree_map(
        _make_replacement, pytree, is_leaf=lambda x: x in node_replace_fns
    )

    # Check that `where` is well-formed.
    for node in nodes:
        count = node_replace_fns.count(node)
        if count == 0:
            raise ValueError(
                "`where` does not specify an element or elements of `pytree`."
            )
        elif count == 1:
            pass
        else:
            raise ValueError(
                "`where` does not uniquely identify a single element of `pytree`. This "
                "usually occurs when trying to replace a `None` value:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True)\n"
                "\n"
                "\n"
                "for which the fix is to specify that `None`s should be treated as "
                "leaves:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True,\n"
                "  ...             is_leaf=lambda x: x is None)"
            )
    out = optree.tree_map(
        lambda x: () if isinstance(x, _DistinctTuple) else x,
        out,
        is_leaf=lambda x: isinstance(x, _DistinctTuple),
    )
    return out

pt = optree.pytree.reexport(namespace="hepstats_pytree_utils")

@pt.register_node_class(namespace="hepstats_pytree_utils")
class PVList(list):
    def __init__(self, iterable, to_original=None):
        if to_original is not None:
            self._to_original = to_original
        elif isinstance(iterable, list) or isinstance(iterable, tuple):
            self._to_original = type(iterable)
        elif hasattr(iterable, "__array__"):
            self._to_original = np.array
        elif not isinstance(iterable, Iterable):
            self._to_original = lambda x: x[0]
            iterable = [iterable]
        else:
            raise ValueError("Unsupported type in {}".format(type(iterable)))
            
        super().__init__(iterable)
        
    def tree_flatten(self):
        return (list(self), self._to_original, None)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children, to_original=metadata)

    def original(self):
        return self._to_original(self)

    @property
    def to_original(self):
        return to_original

    def new_all(self, value):
        return PVList([value] * len(self), to_original=self._to_original)

    # def map(self, func):
    #     for i in range(len(self)):
    #         self[i] = func(self[i])

    def __repr__(self):
        return "PVList({})".format(super().__repr__())

pt.PyTree = PyTree
pt.ravel = tree_ravel

pt.at = tree_at
# pt.PVList = PVList