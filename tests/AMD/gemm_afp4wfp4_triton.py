# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py
# Kernels were copypasted and output pointer was moved to the last position.
# The launch function was rewritten to use JAX primitives and match jax-triton calling
# conventions

import functools
from typing import Optional, Tuple

from strided_array import StridedArray

import triton
import triton.language as tl

import _aiter
import arch_info
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.heuristics({
  "EVEN_K": lambda args: (
    (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
    and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
    and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0)
  ),
})
@triton.jit
def _gemm_afp4wfp4_kernel(
  a_ptr,
  b_ptr,
  # c_ptr,
  a_scales_ptr,
  b_scales_ptr,
  M,
  N,
  K,
  stride_am,
  stride_ak,
  stride_bk,
  stride_bn,
  stride_ck,
  stride_cm,
  stride_cn,
  stride_asm,
  stride_ask,
  stride_bsn,
  stride_bsk,
  c_ptr,  # output goes last
  # Meta-parameters
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_K: tl.constexpr,
  GROUP_SIZE_M: tl.constexpr,
  NUM_KSPLIT: tl.constexpr,
  SPLITK_BLOCK_SIZE: tl.constexpr,
  EVEN_K: tl.constexpr,
  num_warps: tl.constexpr,
  num_stages: tl.constexpr,
  waves_per_eu: tl.constexpr,
  matrix_instr_nonkdim: tl.constexpr,
  cache_modifier: tl.constexpr,
):
  """
  Kernel for computing the matmul C = A x B.
  A and B inputs are in the microscale fp4 (mxfp4) format.
  A_scales and B_scales are in e8m0 format.
  A has shape (M, K), B has shape (K, N) and C has shape (M, N)
  """

  tl.assume(stride_am > 0)
  tl.assume(stride_ak > 0)
  tl.assume(stride_bk > 0)
  tl.assume(stride_bn > 0)
  tl.assume(stride_cm > 0)
  tl.assume(stride_cn > 0)
  tl.assume(stride_asm > 0)
  tl.assume(stride_ask > 0)
  tl.assume(stride_bsk > 0)
  tl.assume(stride_bsn > 0)

  GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

  # -----------------------------------------------------------
  # Map program ids `pid` to the block of C it should compute.
  # This is done in a grouped ordering to promote L2 data reuse.
  pid_unified = tl.program_id(axis=0)
  # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
  pid_unified = _aiter.remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

  pid_k = pid_unified % NUM_KSPLIT
  pid = pid_unified // NUM_KSPLIT
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

  if NUM_KSPLIT == 1:
    pid_m, pid_n = _aiter.pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
  else:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

  tl.assume(pid_m >= 0)
  tl.assume(pid_n >= 0)
  tl.assume(pid_k >= 0)
  # We assume 32 elements along K share the same scale.
  SCALE_GROUP_SIZE: tl.constexpr = 32

  if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
    num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # Create pointers for the first block of A and B scales
    offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
      0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
    )
    a_scale_ptrs = (
      a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    )
    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (
      b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
      # When EVEN_K is true, K / SG is an exact multiple of (BLOCK_SIZE_K / SG),
      # so the scale-pointer arithmetic stays in-bounds and the unconditional
      # tl.load matches the register layout tl.dot_scaled expects.
      # When EVEN_K is false, the tail K-iteration overshoots the scale buffer
      # in the K-scale axis. We mask out those lanes to avoid reading 0xFF bytes
      # (which decode as e8m0 NaN) from past-the-end memory.
      if EVEN_K:
        a_scales = tl.load(a_scale_ptrs)
        b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)
      else:
        k_scale_mask = offs_ks < (2 * K // SCALE_GROUP_SIZE) - k * (
          BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scales = tl.load(a_scale_ptrs, mask=k_scale_mask[None, :], other=0)
        b_scales = tl.load(
          b_scale_ptrs,
          mask=k_scale_mask[None, :],
          other=0,
          cache_modifier=cache_modifier,
        )
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0)
        b = tl.load(
          b_ptrs,
          mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
          other=0,
          cache_modifier=cache_modifier,
        )

      accumulator = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator)

      # Advance the ptrs to the next K block.
      a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
      b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
      a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
      b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = (
      c_ptr
      + stride_cm * offs_cm[:, None]
      + stride_cn * offs_cn[None, :]
      + pid_k * stride_ck
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _gemm_afp4wfp4_reduce_kernel(
  c_in_ptr,
  # c_out_ptr,
  M,
  N,
  stride_c_in_k,
  stride_c_in_m,
  stride_c_in_n,
  stride_c_out_m,
  stride_c_out_n,
  c_out_ptr,  # output goes last
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
  ACTUAL_KSPLIT: tl.constexpr,
  MAX_KSPLIT: tl.constexpr,
):

  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)

  offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, MAX_KSPLIT)
  c_in_ptrs = (
    c_in_ptr
    + (offs_k[:, None, None] * stride_c_in_k)
    + (offs_m[None, :, None] * stride_c_in_m)
    + (offs_n[None, None, :] * stride_c_in_n)
  )

  if ACTUAL_KSPLIT == MAX_KSPLIT:
    c = tl.load(c_in_ptrs)
  else:
    c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
  c = tl.sum(c, axis=0)

  c = c.to(c_out_ptr.type.element_ty)

  c_out_ptrs = (
    c_out_ptr + (offs_m[:, None] * stride_c_out_m) + (offs_n[None, :] * stride_c_out_n)
  )

  tl.store(c_out_ptrs, c)


def _get_config(
  M: int,
  N: int,
  K: int,
  shuffle: bool = False,
):
  # Note: Config files use K=2*K in their naming
  K = 2 * K
  if shuffle:
    return _aiter.get_gemm_config("GEMM-AFP4WFP4_PRESHUFFLED", M, N, K)
  else:
    return _aiter.get_gemm_config("GEMM-AFP4WFP4", M, N, K)


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
  # heuristics for make "EVEN_K == True" as much as possible
  NUM_KSPLIT_STEP = 2
  BLOCK_SIZE_K_STEP = 2
  SPLITK_BLOCK_SIZE = (
    triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
  )
  while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
    if (
      K % (SPLITK_BLOCK_SIZE // 2) == 0
      and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
      and K % (BLOCK_SIZE_K // 2) == 0
    ):
      break
    elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
      NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
    elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
      if NUM_KSPLIT > 1:
        NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
      elif BLOCK_SIZE_K > 16:
        BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
    elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
      BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
    else:
      break

    SPLITK_BLOCK_SIZE = (
      triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )

  # re-ensuring NUM_KSPLIT is the correct value
  NUM_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE // 2))

  return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def gemm_afp4wfp4(
  x: StridedArray,
  w: StridedArray,
  x_scales: StridedArray,
  w_scales: StridedArray,
  dtype: Optional[jnp.dtype] = jnp.bfloat16,
  config: Optional[dict] = None,
  skip_reduce: Optional[bool] = False,
) -> jnp.ndarray:
  """Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.

  This launcher trusts its inputs: it neither inspects the underlying buffer
  orientation nor copies any tensor for layout reasons. The caller is responsible
  for delivering each operand with the strides expected by the kernel below;
  precondition assertions in the launcher will fail loud if that contract is broken.

  The only stride-affecting operation the launcher performs is the ``w = w.T``
  that has always been part of its API contract; this is implemented as a
  zero-copy Python tuple swap on the wrapper's strides before they are passed
  into the private jitted inner.

  Per-operand layout precondition (on the wrapper's logical view, with strides in
  element units):

      x.shape          == (M, K // 2)             ; x.strides        == (K // 2, 1)
      w.shape          == (N, K // 2)             ; w.strides        == (K // 2, 1)   (pre-T)
      x_scales.shape   == (M, K // SCALE_GROUP_SIZE)
                                                  ; x_scales.strides[0] == 1
      w_scales.shape   == (N, K // SCALE_GROUP_SIZE)
                                                  ; w_scales.strides[0] == 1

  After the internal w = w.T the kernel sees w with shape (K // 2, N) and strides
  (1, K // 2), i.e. K unit-stride. The kernel-tile register layouts (blocked_mk,
  blocked_kn, blocked_scales) are tuned for exactly these orientations.

  Args:
      x: FP4 E2M1 activation matrix wrapper, logical shape (M, K // 2),
          K unit-stride.
      w: FP4 E2M1 weight matrix wrapper, logical shape (N, K // 2), K unit-stride.
          Internally flipped to (K // 2, N) before the kernel call via a stride
          tuple swap.
      x_scales: E8M0 per-group scales for x, logical shape (M, K // 32),
          M unit-stride. One scale per 32 elements in the K dimension.
      w_scales: E8M0 per-group scales for w, logical shape (N, K // 32),
          N unit-stride.
      dtype: Output dtype (jnp.bfloat16 or jnp.float16).
      config: Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
          BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
      skip_reduce: If True and NUM_KSPLIT > 1, returns the partial-reduction
          tensor of shape (NUM_KSPLIT, M, N).

  Returns:
      The output tensor with shape (M, N), or (NUM_KSPLIT, M, N) when
      ``skip_reduce`` is True with NUM_KSPLIT > 1.

  Raises:
      AssertionError: If any input violates the layout precondition.
  """
  assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

  M, K2 = x.shape
  N, K2_w = w.shape
  assert K2 == K2_w, f"K mismatch: x.shape[1]={K2}, w.shape[1]={K2_w}"
  K = K2 * 2

  assert x.strides == (K2, 1), f"x.strides={x.strides}; expected ({K2}, 1)"
  assert w.strides == (K2, 1), f"w.strides={w.strides}; expected ({K2}, 1) (pre-T)"
  assert x_scales.shape == (M, K // 32)
  assert w_scales.shape == (N, K // 32)
  assert x_scales.strides[0] == 1, (
    f"x_scales.strides={x_scales.strides}; expected strides[0]==1"
  )
  assert w_scales.strides[0] == 1, (
    f"w_scales.strides={w_scales.strides}; expected strides[0]==1"
  )

  w_strides_post_T = (w.strides[1], w.strides[0])

  return _gemm_afp4wfp4_jit(
    x.data,
    w.data,
    x_scales.data,
    w_scales.data,
    x_strides=x.strides,
    w_strides=w_strides_post_T,
    x_scales_strides=x_scales.strides,
    w_scales_strides=w_scales.strides,
    dtype=dtype,
    config=config,
    skip_reduce=skip_reduce,
  )


@functools.partial(
  jax.jit,
  static_argnames=(
    "x_strides",
    "w_strides",
    "x_scales_strides",
    "w_scales_strides",
    "dtype",
    "config",
    "skip_reduce",
  ),
)
def _gemm_afp4wfp4_jit(
  x_data: jnp.ndarray,
  w_data: jnp.ndarray,
  x_scales_data: jnp.ndarray,
  w_scales_data: jnp.ndarray,
  *,
  x_strides: Tuple[int, int],
  w_strides: Tuple[int, int],
  x_scales_strides: Tuple[int, int],
  w_scales_strides: Tuple[int, int],
  dtype: jnp.dtype,
  config: Optional[dict],
  skip_reduce: bool,
) -> jnp.ndarray:
  """Private jitted launcher; layout contract validated by gemm_afp4wfp4."""

  M, K = x_data.shape
  N = w_data.shape[0]

  if config is None:
    config, _ = _get_config(M, N, K)

  assert config["NUM_KSPLIT"] >= 1

  if config["NUM_KSPLIT"] > 1:
    SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
      K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
    )

    config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
    config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
    config["NUM_KSPLIT"] = NUM_KSPLIT

  if config["BLOCK_SIZE_K"] >= 2 * K:
    config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
    config["SPLITK_BLOCK_SIZE"] = 2 * K
    config["NUM_KSPLIT"] = 1
  config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

  unit_NUM_KSPLIT = config["NUM_KSPLIT"] == 1
  return_y_pp = not unit_NUM_KSPLIT and skip_reduce

  if not unit_NUM_KSPLIT:
    y_pp_shape = (config["NUM_KSPLIT"], M, N)
    y_pp_dtype = dtype if _USE_GEMM_SPLITK_BF16 else jnp.float32
    y_pp_stride = (y_pp_shape[1] * y_pp_shape[2], y_pp_shape[2], 1)
  else:
    config["SPLITK_BLOCK_SIZE"] = 2 * K
    y_pp, y_pp_stride, y_pp_shape, y_pp_dtype = None, None, None, None

  y_shape = (M, N)

  # config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

  grid = lambda META: (  # noqa: E731
    (
      META["NUM_KSPLIT"]
      * triton.cdiv(M, META["BLOCK_SIZE_M"])
      * triton.cdiv(N, META["BLOCK_SIZE_N"])
    ),
  )

  if unit_NUM_KSPLIT:
    out_tensor_y_shape = y_shape
    out_tensor_y_dtype = dtype
  else:
    out_tensor_y_shape = y_pp_shape
    out_tensor_y_dtype = y_pp_dtype

  result = jt.triton_call(
    x_data,
    w_data,
    x_scales_data,
    w_scales_data,
    M,
    N,
    K,
    x_strides[0],
    x_strides[1],
    w_strides[0],
    w_strides[1],
    0 if unit_NUM_KSPLIT else y_pp_stride[0],
    y_shape[1] if unit_NUM_KSPLIT else y_pp_stride[1],
    1 if unit_NUM_KSPLIT else y_pp_stride[2],
    x_scales_strides[0],
    x_scales_strides[1],
    w_scales_strides[0],
    w_scales_strides[1],
    kernel=_gemm_afp4wfp4_kernel,
    out_shape=jax.ShapeDtypeStruct(shape=out_tensor_y_shape, dtype=out_tensor_y_dtype),
    grid=grid,
    **config,
  )

  if return_y_pp:
    return result
  elif not unit_NUM_KSPLIT:
    y_pp = result
    REDUCE_BLOCK_SIZE_M = 16
    # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
    # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
    # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
    REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
    ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

    grid_reduce = (
      triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
      triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
    )

    y = jt.triton_call(
      y_pp,
      M,
      N,
      y_pp_stride[0],
      y_pp_stride[1],
      y_pp_stride[2],
      y_shape[1],  # y.stride(0),
      1,  # y.stride(1),
      kernel=_gemm_afp4wfp4_reduce_kernel,
      out_shape=jax.ShapeDtypeStruct(shape=y_shape, dtype=dtype),
      grid=grid_reduce,
      BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
      BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
      ACTUAL_KSPLIT=ACTUAL_KSPLIT,
      MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
    )

  else:
    y = result

  return y


def gemm_afp4wfp4_from_arrays(x, w, x_scales, w_scales, *args, **kwargs):
  """Convenience adapter for callers that have raw jnp.ndarray.

  The same layout contract documented on gemm_afp4wfp4 still applies; this helper
  only saves the caller from writing StridedArray.from_array four times. No data
  is copied. If the raw arrays are not in the kernel-optimal orientation, the
  launcher's contract assertions will fail.
  """
  return gemm_afp4wfp4(
    StridedArray.from_array(x),
    StridedArray.from_array(w),
    StridedArray.from_array(x_scales),
    StridedArray.from_array(w_scales),
    *args,
    **kwargs,
  )
