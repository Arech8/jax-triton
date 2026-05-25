"""Plain-Python strided view over a contiguous jax.Array (test driver helper)."""

from __future__ import annotations

from typing import Tuple, Union

import jax
import jax.numpy as jnp

ShapeT = Tuple[int, ...]
StrideT = Tuple[int, ...]

IndexKey = Union[int, slice, Tuple[Union[int, slice], ...]]


def default_strides(shape: ShapeT) -> StrideT:
  if not shape:
    return ()
  stride = 1
  strides: list[int] = []
  for size in reversed(shape):
    strides.insert(0, stride)
    stride *= size
  return tuple(strides)


class StridedArray:
  """Zero-copy stride/view wrapper over a C-contiguous jax.Array."""

  data: jax.Array
  shape: ShapeT
  strides: StrideT
  offset: int

  def __init__(
    self,
    data: jax.Array,
    shape: ShapeT | None = None,
    strides: StrideT | None = None,
    offset: int = 0,
  ):
    if shape is None:
      shape = tuple(data.shape)
    if strides is None:
      strides = default_strides(shape)
    if len(shape) != len(strides):
      raise ValueError(
        f"shape ndim {len(shape)} != strides ndim {len(strides)}"
      )
    if offset < 0:
      raise ValueError(f"offset must be non-negative, got {offset}")
    for s in strides:
      if s < 0:
        raise ValueError(f"strides must be non-negative, got {strides}")

    self.data = data
    self.shape = tuple(shape)
    self.strides = tuple(strides)
    self.offset = offset
    self._validate_bounds()

  def _validate_bounds(self) -> None:
    if self.ndim == 0:
      if self.offset >= self.data.size:
        raise ValueError("offset out of bounds for scalar view")
      return
    max_linear = self.offset
    for dim, (size, stride) in enumerate(zip(self.shape, self.strides)):
      if size < 0:
        raise ValueError(f"shape[{dim}] must be non-negative, got {size}")
      if size > 0:
        max_linear += (size - 1) * stride
    if max_linear >= self.data.size:
      raise ValueError(
        f"view exceeds data.size={self.data.size} (max linear index {max_linear})"
      )

  @classmethod
  def from_array(cls, data: jax.Array) -> StridedArray:
    shape = tuple(data.shape)
    return cls(data, shape=shape, strides=default_strides(shape), offset=0)

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def stride(self, axis: int) -> int:
    if axis < 0:
      axis += self.ndim
    if axis < 0 or axis >= self.ndim:
      raise IndexError(f"axis {axis} out of range for ndim {self.ndim}")
    return self.strides[axis]

  @property
  def dtype(self) -> jnp.dtype:
    return self.data.dtype

  @property
  def T(self) -> StridedArray:
    if self.ndim != 2:
      raise NotImplementedError("T is only defined for 2-D arrays")
    return self.transpose(1, 0)

  def transpose(self, *axes: int) -> StridedArray:
    if len(axes) != self.ndim:
      raise ValueError(
        f"transpose expects {self.ndim} axes, got {len(axes)}"
      )
    return StridedArray(
      self.data,
      shape=tuple(self.shape[i] for i in axes),
      strides=tuple(self.strides[i] for i in axes),
      offset=self.offset,
    )

  def __getitem__(self, key: IndexKey) -> StridedArray:
    if isinstance(key, tuple):
      if len(key) != self.ndim:
        raise NotImplementedError(
          f"tuple indexing with {len(key)} entries for ndim {self.ndim}"
        )
      result = self
      for axis, k in enumerate(key):
        result = result._index_axis(axis, k)
      return result

    return self._index_axis(0, key)

  def _index_axis(self, axis: int, key: Union[int, slice]) -> StridedArray:
    if axis != 0:
      raise NotImplementedError("only outer-axis indexing is supported")

    if isinstance(key, slice):
      if key.start not in (None, 0) or key.step not in (None, 1):
        raise NotImplementedError(
          "only slices of the form :n (start=0, step=1) are supported"
        )
      stop = key.stop
      if stop is None:
        stop = self.shape[0]
      if stop < 0:
        raise ValueError(f"slice stop must be non-negative, got {stop}")
      new_shape = (stop,) + self.shape[1:]
      return StridedArray(
        self.data,
        shape=new_shape,
        strides=self.strides,
        offset=self.offset,
      )

    if isinstance(key, int):
      if key < 0:
        key += self.shape[0]
      if key < 0 or key >= self.shape[0]:
        raise IndexError(f"index {key} out of range for axis size {self.shape[0]}")
      new_offset = self.offset + key * self.strides[0]
      new_shape = self.shape[1:]
      new_strides = self.strides[1:]
      return StridedArray(
        self.data,
        shape=new_shape,
        strides=new_strides,
        offset=new_offset,
      )

    raise NotImplementedError(f"unsupported index type {type(key)!r}")

  def _view_as_permutation_and_slice(
    self,
  ) -> tuple[tuple[int, ...], tuple[slice, ...]]:
    if self.offset != 0:
      raise NotImplementedError("to_jax with non-zero offset is not supported")
    if self.ndim != 2:
      raise NotImplementedError("to_jax is only implemented for 2-D views")

    d0, d1 = self.data.shape
    s0, s1 = self.shape
    st0, st1 = self.strides

    if st1 == 1 and st0 == d1:
      perm = (0, 1)
    elif st0 == 1 and st1 == d1:
      perm = (1, 0)
    else:
      raise NotImplementedError(
        f"unsupported 2-D stride pattern strides={self.strides} "
        f"for data.shape={self.data.shape}"
      )

    return perm, (slice(0, s0), slice(0, s1))

  def to_jax(self) -> jax.Array:
    if (
      self.offset == 0
      and self.shape == tuple(self.data.shape)
      and self.strides == default_strides(self.shape)
    ):
      return self.data

    if self.offset == 0 and self.ndim == 2 and self.strides == (1, 1):
      return jnp.reshape(self.data, self.shape)

    perm, post_slice = self._view_as_permutation_and_slice()
    arr = jnp.transpose(self.data, perm)
    return arr[post_slice]


# Register StridedArray as a JAX pytree below isn't mandatory, but simplifies, for
# example benchmarking, that doesn't have to handle special case before calling
# `jax.block_until_ready()` on data elements returned by an input generating function
# in form of StridedArray objects.
def _strided_flatten(sa):
    children = (sa.data,)
    aux = (sa.shape, sa.strides, sa.offset)
    return children, aux

def _strided_unflatten(aux, children):
    shape, strides, offset = aux
    (data,) = children
    return StridedArray(data, shape=shape, strides=strides, offset=offset)

jax.tree_util.register_pytree_node(StridedArray, _strided_flatten, _strided_unflatten)