import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config, random, tree_util
import jax.numpy as jnp
import jax_triton as jt
import jax_triton.triton_lib as jttl
import numpy as np
import triton
import triton.language as tl
import types

config.parse_flags_with_absl()


def setUpModule():
  config.update("jax_enable_x64", True)


def tearDownModule():
  config.update("jax_enable_x64", False)


class TritonCallTupleTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2, 3, 4)
  def test_index(self, size):
    self.skipTest("TODO")
    return
    @triton.jit
    def _tuple_increment(values):
      return tl.tuple([v + 1 for v in values])

    @triton.jit
    def _tuple_index_func(Ptrs, values):
      for i in tl.static_range(len(values)):
        tl.store(Ptrs[i], values[i])

    @jt.kernel(out_names="Ptrs", input_output_aliases={1: 0})
    @triton.jit
    def _tuple_index(_0, Ptrs, _1: tl.constexpr, values, _2, _3: tl.constexpr, _4):
      values = _tuple_increment(values)
      _tuple_index_func(Ptrs, values)

    """
    def tuple_index(vals):
        rets = tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in vals)
        # TODO also try to call as positional args only as in the original
        # _tuple_index[(1, )](0, rets, 0, vals, 0, 0, 0)
        return jt.triton_call(
            Ptrs=rets,
            values=vals,
            _0=0,
            _1=0,
            _2=0,
            _3=0,
            _4=0,
            kernel=_tuple_index,
            out_shape={"Ptrs": rets},
            grid=(1,),
            input_output_aliases={0: 0},
        )
        #TODO this requires (1) unpacking tuples in out_shape. (2) fixing
        # input_output_aliases correspondingly to point to the tuple values, (3)
        # fixing zeroed_outputs handling, that requires bijection for input_output_aliases
    """

    vals = tuple(i + 1 for i in range(size))
    rets = tuple(jnp.zeros((1,), dtype=jnp.float32) for _ in vals)
    rets = _tuple_index[(1, )](0, rets, 0, vals, 0, 0, 0, out_shape=rets)

    # _tuple_index[(1,)](0, rets, 0, vals, 0, 0, 0)
    assert vals == tuple(x - 1 for x in rets)


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
