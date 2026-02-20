import sys

# from absl.testing import absltest
# from absl.testing import parameterized
import pytest

import jax
import jax.numpy as jnp
#from jax import config
from jax import random
import jax_triton as jt
import numpy as np
import triton

# import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
# from triton.language.extra import libdevice

from generic import get_arch, is_fp4_avail
from gemm_afp4wfp4_gluon import gemm_afp4wfp4 as gluon_gemm_afp4wfp4

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
):
  assert not isinstance(dtype, str)

  k = random.key(5)

  if layout[0] == "T":
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = random.randint(k, (M, K // 2), 0, 16, dtype=jnp.uint8)
    x_high = random.randint(k, (M, K // 2), 0, 16, dtype=jnp.uint8)
  else:
    x_low = random.randint(k, (K // 2, M), 0, 16, dtype=jnp.uint8).T
    x_high = random.randint(k, (K // 2, M), 0, 16, dtype=jnp.uint8).T

  if layout[1] == "N":
    w_low = random.randint(k, (N, K // 2), 0, 16, dtype=jnp.uint8)
    w_high = random.randint(k, (N, K // 2), 0, 16, dtype=jnp.uint8)
  else:
    w_low = random.randint(k, (K // 2, N), 0, 16, dtype=jnp.uint8).T
    w_high = random.randint(k, (K // 2, N), 0, 16, dtype=jnp.uint8).T

  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
  x = x_high << 4 | x_low
  # x = x.to(device="cuda")

  w = w_low | w_high << 4
  # Scale of 1.0 in e8m0, bias 127.
  M_pad = (M + 255) // 256 * 256
  x_scales = random.randint(
    k, (K // SCALE_GROUP_SIZE, M_pad), 124, 128, dtype=jnp.uint8
  )
  w_scales = random.randint(k, (K // SCALE_GROUP_SIZE, N), 124, 128, dtype=jnp.uint8)
  x_scales = x_scales.T
  w_scales = w_scales.T

  x_scales_shuffled = x_scales
  w_scales_shuffled = w_scales

  w_shuffed = w

  y = None
  if output:
    y = jnp.empty((M, N), dtype=dtype)
    out_dtype = (None,)
  else:
    out_dtype = dtype

  return (
    x,
    w,
    w_shuffed,
    x_scales[:M],
    w_scales,
    x_scales_shuffled[:M],
    w_scales_shuffled,
    out_dtype,
    y,
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
  # return x_vals
  return [(128, 1280, 8192)]


def mxfp4_to_f32(x):
  # 2 because we pack fp4 in uint8.
  # x = x.repeat_interleave(2, dim=1)
  x = jnp.repeat(x, 2, axis=1)
  # x[:, ::2] = x[:, ::2] & 0xF
  x = x.at[:, ::2].set(x[:, ::2] & 0xF)
  # x[:, 1::2] = x[:, 1::2] >> 4
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
  x_f32 = 2 ** ((x - 127).astype(jnp.float32))
  # x_f32[x_f32 == 128] = float("nan")
  x_f32 = x_f32.at[x_f32 == 128].set(jnp.nan)
  return x_f32


def jax_afp4wfp4(x, w, x_scales, w_scales, dtype):
  # First convert the x and w inputs to f32.
  x_f32 = mxfp4_to_f32(x)
  w_f32 = mxfp4_to_f32(w)
  # Next convert the e8m0 scales to f32.

  # x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
  x_scales = jnp.repeat(x_scales, SCALE_GROUP_SIZE, axis=1).astype(jnp.float32)

  x_scales_f32 = e8m0_to_f32(x_scales)
  x_f32 = x_f32 * x_scales_f32
  # w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
  w_scales = jnp.repeat(w_scales, SCALE_GROUP_SIZE, axis=1).astype(jnp.float32)
  w_scales_f32 = e8m0_to_f32(w_scales)
  w_f32 = w_f32 * w_scales_f32
  # return torch.mm(x_f32, w_f32.T).to(dtype)
  return jnp.matmul(x_f32, w_f32.T).astype(dtype)


def run_triton(
  x, w, x_scales, w_scales, dtype=jnp.bfloat16, y=None, skip_reduce=False, impl=None
):
  return impl(x, w, x_scales, w_scales, dtype, y, skip_reduce=skip_reduce)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
# @pytest.mark.parametrize(
#  "shuffle_weight_scales",
#  [True, False],
# )
@pytest.mark.parametrize("skip_reduce", [True, False])
# @pytest.mark.parametrize("impl", ["triton", "gluon"])
@pytest.mark.parametrize("impl", ["gluon"])
def test_gemm_afp4_wfp4(
  M: int,
  N: int,
  K: int,
  dtype,
  layout,
  output,
  # shuffle_weight_scales,
  skip_reduce,
  impl,
):

  if not is_fp4_avail():
    pytest.skip("MXFP4 not supported on this architecture (requires CDNA4).")

  (
    x,
    w,
    w_triton,
    x_scales,
    w_scales,
    x_scales_triton,
    w_scales_triton,
    out_dtype,
    y,
  ) = generate_gemm_afp4wfp4_inputs(
    M,
    N,
    K,
    dtype,
    layout=layout,
    output=output,
  )

  expected = jax_afp4wfp4(x, w, x_scales, w_scales, dtype)

  # if impl == "triton":
  #  impl = triton_gemm_afp4wfp4
  # elif impl == "gluon":
  if impl == "gluon":
    impl = gluon_gemm_afp4wfp4
  else:
    raise ValueError(f"Unknown implementation: {impl}")

  if output:
    triton_out = run_triton(
      x,
      w_triton,
      x_scales_triton,
      w_scales_triton,
      dtype,
      y,
      skip_reduce=skip_reduce,
      impl=impl,
    )
  else:
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
  sys.exit(pytest.main([__file__]))
