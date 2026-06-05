# adapted from https://github.com/ROCm/gfx950-gluon-tutorials/blob/7fc24ef0cc49b4c99036482a8174292564d5a48b/kernels/gemm/a4w4/bench.py#

##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

import sys

import pytest
import jax.numpy as jnp

# from jax import random
import numpy as np
import arch_info

from gemm_a4w4_from_gluon_tutorial import matmul as a4w4_gemm
from gemm_afp4wfp4_test import (
  mxfp4_to_f32,
  e8m0_to_f32,
  generate_gemm_afp4wfp4_inputs,
  jax_afp4wfp4,
)


def generate_mxfp4_inputs(M, N, K):
  """Generate random MXFP4 packed tensors and e8m0 scales.

  Returns:
      a_fp4: (M, K//2) uint8 — packed MXFP4 activations
      b_fp4: (N, K//2) uint8 — packed MXFP4 weights (N,K//2 layout)
      a_scales: (M, K//32) uint8 — e8m0 scales for A
      b_scales: (N, K//32) uint8 — e8m0 scales for B
  """
  a_fp4, b_fp4, _, a_scales, b_scales, _, _, _, _ = generate_gemm_afp4wfp4_inputs(
    M, N, K, dtype=jnp.bfloat16, layout="TN", output=False
  )
  return a_fp4, b_fp4, a_scales, b_scales


def get_x_vals():
  return [
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 3072),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 4096, 16384),
    (4096, 4096, 32768),
  ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_gemm_afp4_wfp4(
  M: int,
  N: int,
  K: int,
):
  if not arch_info.is_fp4_avail():
    pytest.skip("MXFP4 not supported on this architecture (requires CDNA4).")

  dtype = jnp.bfloat16

  (
    x,
    w,
    w_triton,
    x_scales,
    w_scales,
    x_scales_triton,
    w_scales_triton,
    _,
    _,
  ) = generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout="TN", output=False)

  expected = jax_afp4wfp4(
    x.to_jax(),
    w.to_jax(),
    x_scales.to_jax(),
    w_scales.to_jax(),
    dtype,
  )

  triton_out = a4w4_gemm(x, w_triton, x_scales_triton, w_scales_triton)

  if triton_out.ndim == 3:
    triton_out = triton_out.sum(axis=0).astype(dtype)

  atol = 1e-1
  rtol = 0

  np.testing.assert_allclose(triton_out, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
  sys.exit(pytest.main(sys.argv))
