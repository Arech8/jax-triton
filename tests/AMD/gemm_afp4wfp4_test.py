import sys

import pytest

import jax.numpy as jnp
from jax import random
import numpy as np

import arch_info
from gemm_afp4wfp4_gluon import gemm_afp4wfp4 as gluon_gemm_afp4wfp4
from gemm_afp4wfp4_triton import gemm_afp4wfp4 as triton_gemm_afp4wfp4
from strided_array import StridedArray

# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b/aiter/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
  M,
  N,
  K,
  dtype,
  layout="TN",
  output=True,
  key=random.key(5),
):
  assert not isinstance(dtype, str)

  def _randint(key, shape, lo, hi, dtype):
    key, sub = random.split(key)
    return key, random.randint(sub, shape, lo, hi, dtype=dtype)

  if layout[0] == "T":
    key, x_low = _randint(key, (M, K // 2), 0, 16, jnp.uint8)
    key, x_high = _randint(key, (M, K // 2), 0, 16, jnp.uint8)
    x_data = x_high << 4 | x_low
    x = StridedArray.from_array(x_data)
  else:
    key, x_low = _randint(key, (K // 2, M), 0, 16, jnp.uint8)
    key, x_high = _randint(key, (K // 2, M), 0, 16, jnp.uint8)
    x_swapped = x_high << 4 | x_low
    x_data = StridedArray.from_array(x_swapped).T.to_jax()
    x = StridedArray.from_array(x_data)

  if layout[1] == "N":
    key, w_low = _randint(key, (N, K // 2), 0, 16, jnp.uint8)
    key, w_high = _randint(key, (N, K // 2), 0, 16, jnp.uint8)
    w_data = w_low | w_high << 4
    w = StridedArray.from_array(w_data)
  else:
    key, w_low = _randint(key, (K // 2, N), 0, 16, jnp.uint8)
    key, w_high = _randint(key, (K // 2, N), 0, 16, jnp.uint8)
    w_swapped = w_low | w_high << 4
    w_data = StridedArray.from_array(w_swapped).T.to_jax()
    w = StridedArray.from_array(w_data)

  M_pad = (M + 255) // 256 * 256
  key, xs_data = _randint(key, (K // SCALE_GROUP_SIZE, M_pad), 124, 128, jnp.uint8)
  key, ws_data = _randint(key, (K // SCALE_GROUP_SIZE, N), 124, 128, jnp.uint8)

  x_scales = StridedArray.from_array(xs_data).T[:M]
  w_scales = StridedArray.from_array(ws_data).T

  x_scales_shuffled = x_scales
  w_scales_shuffled = w_scales

  w_shuffled = w

  return (
    x,
    w,
    w_shuffled,  # w_triton
    x_scales,  # x_scales
    w_scales,  # w_scales
    x_scales_shuffled,  # x_scales_triton
    w_scales_shuffled,  # w_scales_triton
  )


def get_x_vals():
  x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
  x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
  x_vals += [
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
  ]
  x_vals += [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
  # x_vals = [(128, 1024, 4096)]
  x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
  x_vals += [(256, 3584, 2112)]
  x_vals += [(7, 4608, 7168), (7, 7168, 2304)]
  x_vals += [(v, 106496, 16384) for v in [1, 8, 16, 32, 64, 128, 256]]
  x_vals += [(v, 16384, 53248) for v in [1, 8, 16, 32, 64, 128, 256]]
  x_vals += [(v, 18432, 16384) for v in [1, 8, 16, 32, 64, 128, 256]]
  x_vals += [(v, 16384, 16384) for v in [1, 8, 16, 32, 64, 128, 256]]
  x_vals += [(v, 10240, 8192) for v in [1, 2, 4, 8, 16, 32, 64]]
  x_vals += [(v, 8192, 8192) for v in [1, 2, 4, 8, 16, 32, 64]]
  x_vals += [(v, 57344, 8192) for v in [1, 2, 4, 8, 16, 32, 64]]
  x_vals += [(v, 8192, 28672) for v in [1, 2, 4, 8, 16, 32, 64]]
  x_vals += [(1, 1, 32)]  # minimal case
  return x_vals
  # return [(128, 1280, 8192)]


def mxfp4_to_f32(x):
  # 2 because we pack fp4 in uint8.
  x = jnp.repeat(x, 2, axis=1)
  x = x.at[:, ::2].set(x[:, ::2] & 0xF)
  x = x.at[:, 1::2].set(x[:, 1::2] >> 4)
  mxfp4_list = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
  ]
  mxfp4_in_f32 = jnp.array(mxfp4_list, dtype=jnp.float32)
  return mxfp4_in_f32[x.astype(jnp.int32)]


def e8m0_to_f32(x):
  x_f32 = 2 ** (x.astype(jnp.float32) - 127)
  # WARNING! The original implementation:
  # x_f32[x_f32 == 128] = float("nan")
  # has a bug: it must map the original x==255 to a NaN, instead it left x_f32 poisoned
  # with inf <= 2**(255-127) == 2**128, but remaps legitimate x==134 or
  # x_f32==2**(134-127)==2**7==128 to NaN
  x_f32 = x_f32.at[x == 255].set(jnp.nan)
  return x_f32


def jax_afp4wfp4(x, w, x_scales, w_scales, dtype):
  # First convert the x and w inputs to f32.
  x_f32 = mxfp4_to_f32(x)
  w_f32 = mxfp4_to_f32(w)
  # Next convert the e8m0 scales to f32.

  x_scales = jnp.repeat(x_scales, SCALE_GROUP_SIZE, axis=1).astype(jnp.float32)

  x_scales_f32 = e8m0_to_f32(x_scales)
  x_f32 = x_f32 * x_scales_f32
  w_scales = jnp.repeat(w_scales, SCALE_GROUP_SIZE, axis=1).astype(jnp.float32)
  w_scales_f32 = e8m0_to_f32(w_scales)
  w_f32 = w_f32 * w_scales_f32
  return jnp.matmul(x_f32, w_f32.T).astype(dtype)


def run_triton(
  x, w, x_scales, w_scales, dtype=jnp.bfloat16, skip_reduce=False, impl=None
):
  return impl(x, w, x_scales, w_scales, dtype, skip_reduce=skip_reduce)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("shuffle_weight_scales", [False])  #  [True, False], )
@pytest.mark.parametrize("skip_reduce", [True, False])
@pytest.mark.parametrize("impl", ["triton", "gluon"])
def test_gemm_afp4_wfp4(
  M: int,
  N: int,
  K: int,
  dtype,
  layout,
  output,
  shuffle_weight_scales,
  skip_reduce,
  impl,
):
  del shuffle_weight_scales  # unused. Left for compatibility with the original test
  if not arch_info.is_fp4_avail():
    pytest.skip("MXFP4 not supported on this architecture (requires CDNA4).")

  (
    x,
    w,
    w_triton,
    x_scales,
    w_scales,
    x_scales_triton,
    w_scales_triton,
  ) = generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout=layout, output=output)

  expected = jax_afp4wfp4(
    x.to_jax(),
    w.to_jax(),
    x_scales.to_jax(),
    w_scales.to_jax(),
    dtype,
  )

  if impl == "triton":
    impl = triton_gemm_afp4wfp4
  elif impl == "gluon":
    impl = gluon_gemm_afp4wfp4
  else:
    raise ValueError(f"Unknown implementation: {impl}")

  triton_out = run_triton(
    x,
    w_triton,
    x_scales_triton,
    w_scales_triton,
    dtype,
    skip_reduce=skip_reduce,
    impl=impl,
  )

  if triton_out.ndim == 3:
    triton_out = triton_out.sum(axis=0).astype(dtype)

  # torch.testing.assert_close(torch_out, triton_out)
  # https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
  rtol = {jnp.float16: 1e-3, jnp.bfloat16: 1.6e-2}
  atol = {jnp.float16: 1e-5, jnp.bfloat16: 1e-5}

  np.testing.assert_allclose(triton_out, expected, rtol=rtol[dtype], atol=atol[dtype])


if __name__ == "__main__":
  sys.exit(pytest.main(sys.argv))
