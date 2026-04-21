# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py
# Kernels are copypasted, the launch function was rewritten to use JAX primitives and
# match jax-triton calling conventions

import functools
from typing import Optional

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
  c_ptr,
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
      a_scales = tl.load(a_scale_ptrs)
      b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)

      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      if EVEN_K:
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)
      else:
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
  c_out_ptr,
  M,
  N,
  stride_c_in_k,
  stride_c_in_m,
  stride_c_in_n,
  stride_c_out_m,
  stride_c_out_n,
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


@functools.partial(
  jax.jit, static_argnames=("dtype", "skip_reduce", "config"), donate_argnames="y"
)
def gemm_afp4wfp4(
  x: jnp.ndarray,
  w: jnp.ndarray,
  x_scales: jnp.ndarray,
  w_scales: jnp.ndarray,
  dtype: Optional[jnp.dtype] = jnp.bfloat16,
  y: Optional[jnp.ndarray] = None,
  config: Optional[dict] = None,
  skip_reduce: Optional[bool] = False,
) -> jnp.ndarray:
  """
  Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.

  Args:
      x (jnp.ndarray): FP4 E2M1 input matrix with shape (M, K//2).
      w (jnp.ndarray): FP4 E2M1 weight matrix with shape (N, K//2), internally transposed.
      x_scales (jnp.ndarray): E8M0 per-group scale for x with shape (M, K//32).
          One scale per 32 elements in K dimension.
      w_scales (jnp.ndarray): E8M0 per-group scale for w with shape (N, K//32).
          One scale per 32 elements in K dimension.
      dtype (Optional[jnp.dtype]): Output datatype (BF16 or FP16).
      y (Optional[jnp.ndarray]): Pre-allocated output tensor with shape (M, N).
      config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
          BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
      skip_reduce (Optional[bool]): skip reduction, y becomes (SPK, M, N) where SPK is determined by config

  Returns:
      y (jnp.ndarray): Output with shape (M, N) or (SPK, M, N).
  """

  assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

  M, K = x.shape
  N, K = w.shape

  # Transpose w
  w = w.T

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
    if _USE_GEMM_SPLITK_BF16:
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=y.dtype)
    else:
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=jnp.float32)

    y_pp_stride = (y_pp.shape[1] * y_pp.shape[2], y_pp.shape[2], 1)
  else:
    config["SPLITK_BLOCK_SIZE"] = 2 * K
    y_pp = None
    y_pp_stride = None

  if y is None and not return_y_pp:
    y = jnp.empty((M, N), dtype=dtype)

  # config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

  grid = lambda META: (  # noqa: E731
    (
      META["NUM_KSPLIT"]
      * triton.cdiv(M, META["BLOCK_SIZE_M"])
      * triton.cdiv(N, META["BLOCK_SIZE_N"])
    ),
  )

  out_tensor_y = y if unit_NUM_KSPLIT else y_pp

  result = jt.triton_call(
    x,
    w,
    out_tensor_y,
    x_scales,
    w_scales,
    M,
    N,
    K,
    x.shape[1],  # x.stride(0),
    1,  # x.stride(1),
    w.shape[1],  # w.stride(0),
    1,  # w.stride(1),
    0 if unit_NUM_KSPLIT else y_pp_stride[0],
    # y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp_stride[1],
    y.shape[1] if unit_NUM_KSPLIT else y_pp_stride[1],
    # y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp_stride[2],
    1 if unit_NUM_KSPLIT else y_pp_stride[2],
    x_scales.shape[1],  # x_scales.stride(0),
    1,  # x_scales.stride(1),
    w_scales.shape[1],  # w_scales.stride(0),
    1,  # w_scales.stride(1),
    kernel=_gemm_afp4wfp4_kernel,
    input_output_aliases={2: 0},
    out_shape=jax.ShapeDtypeStruct(shape=out_tensor_y.shape, dtype=out_tensor_y.dtype),
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
      y,
      M,
      N,
      y_pp_stride[0],
      y_pp_stride[1],
      y_pp_stride[2],
      y.shape[1],  # y.stride(0),
      1,  # y.stride(1),
      kernel=_gemm_afp4wfp4_reduce_kernel,
      out_shape=jax.ShapeDtypeStruct(shape=y.shape, dtype=y.dtype),
      input_output_aliases={1: 0},
      grid=grid_reduce,
      BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
      BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
      ACTUAL_KSPLIT=ACTUAL_KSPLIT,
      MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
    )

  else:
    y = result

  return y
