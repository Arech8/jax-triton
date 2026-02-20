# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b/aiter/ops/triton/gluon/gemm_a8w8.py
# Kernels are copypasted, launch function was rewritten to use JAX primitives and match
# jax-triton calling conventions

import functools
import json
from typing import Optional

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from _aiter_pid_preprocessing import remap_xcd, pid_grid
from _aiter_generic import AITER_TRITON_CONFIGS_PATH
from generic import get_arch, is_fp4_avail


_USE_GEMM_SPLITK_BF16 = False


# @triton.heuristics(
#    {
#        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
#        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
#        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
#    }
# )
@gluon.jit
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
  BLOCK_SIZE_M: gl.constexpr,
  BLOCK_SIZE_N: gl.constexpr,
  BLOCK_SIZE_K: gl.constexpr,
  GROUP_SIZE_M: gl.constexpr,
  NUM_KSPLIT: gl.constexpr,
  SPLITK_BLOCK_SIZE: gl.constexpr,
  EVEN_K: gl.constexpr,
  num_warps: gl.constexpr,
  num_stages: gl.constexpr,
  waves_per_eu: gl.constexpr,
  matrix_instr_nonkdim: gl.constexpr,
  cache_modifier: gl.constexpr,
):
  """
  Kernel for computing the matmul C = A x B.
  A and B inputs are in the microscale fp4 (mxfp4) format.
  A_scales and B_scales are in e8m0 format.
  A has shape (M, K), B has shape (K, N) and C has shape (M, N)
  """
  GRID_MN = gl.cdiv(M, BLOCK_SIZE_M) * gl.cdiv(N, BLOCK_SIZE_N)

  # -----------------------------------------------------------
  # Map program ids `pid` to the block of C it should compute.
  # This is done in a grouped ordering to promote L2 data reuse.
  pid_unified = gl.program_id(axis=0)
  # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
  pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

  pid_k = pid_unified % NUM_KSPLIT
  pid = pid_unified // NUM_KSPLIT
  num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

  if NUM_KSPLIT == 1:
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
  else:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

  # We assume 32 elements along K share the same scale.
  SCALE_GROUP_SIZE: gl.constexpr = 32

  blocked_mk: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 16],
    threads_per_warp=[8, 8],
    warps_per_cta=[num_warps, 1],
    order=[1, 0],
  )

  blocked_kn: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[16, 1],
    threads_per_warp=[8, 8],
    warps_per_cta=[1, num_warps],
    order=[0, 1],
  )

  blocked_scales: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[4, 1],
    threads_per_warp=[8, 8],
    warps_per_cta=[1, num_warps],
    order=[0, 1],
  )

  linear_as: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 2], [0, 4], [64, 0], [128, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
    warp_bases=[[0, 0], [0, 0], [32, 0]],
    block_bases=[],
    shape=[BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
  )

  linear_bs: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 2], [0, 4], [128, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
    warp_bases=[[32, 0], [64, 0], [0, 0]],
    block_bases=[],
    shape=[BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
  )

  linear_mn: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
    warp_bases=[[0, 32], [0, 64], [32, 0]],
    block_bases=[],
    shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
  )

  shared_a: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16, per_phase=2, max_phase=8, order=[1, 0]
  )

  shared_b: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16, per_phase=2, max_phase=8, order=[0, 1]
  )

  shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
    vec=1, per_phase=1, max_phase=1, order=[0, 1]
  )

  mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4,
    instr_shape=[32, 32, 64],
    transposed=True,
    warps_per_cta=[2, num_warps // 2],
  )

  dot_a_layout: gl.constexpr = gl.DotOperandLayout(
    operand_index=0, parent=mfma_layout, k_width=16
  )
  dot_b_layout: gl.constexpr = gl.DotOperandLayout(
    operand_index=1, parent=mfma_layout, k_width=16
  )

  if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
    num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_ak = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(0, blocked_mk))
    offs_ak_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_ak
    offs_bk = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(1, blocked_kn))
    offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
    offs_am = (
      pid_m * BLOCK_SIZE_M
      + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))
    ) % M
    offs_bn = (
      pid_n * BLOCK_SIZE_N
      + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn))
    ) % N
    offs_a = offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
    offs_b = offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Create pointers for the first block of A and B scales
    offs_ks = gl.arange(
      0,
      BLOCK_SIZE_K // SCALE_GROUP_SIZE,
      layout=gl.SliceLayout(0, blocked_scales),
    )
    offs_ks_split = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + offs_ks
    offs_asm = (
      pid_m * BLOCK_SIZE_M
      + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_scales))
    ) % M
    offs_bsn = (
      pid_n * BLOCK_SIZE_N
      + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, blocked_scales))
    ) % N

    offs_as = offs_asm[:, None] * stride_asm + offs_ks_split[None, :] * stride_ask
    # B scales are N x K even though B operand is K x N.
    offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks_split[None, :] * stride_bsk

    # Create shared memories
    smem_a = gl.allocate_shared_memory(
      a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K // 2], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
      b_ptr.type.element_ty, [BLOCK_SIZE_K // 2, BLOCK_SIZE_N], layout=shared_b
    )

    smem_as = gl.allocate_shared_memory(
      a_scales_ptr.type.element_ty,
      [BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
      layout=shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
      b_scales_ptr.type.element_ty,
      [BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
      layout=shared_scales,
    )

    if EVEN_K:
      a = gl.amd.cdna4.buffer_load(
        ptr=a_ptr,
        offsets=offs_a,
      )
      a_scales = gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=offs_as,
      )
    else:
      a = gl.amd.cdna4.buffer_load(
        ptr=a_ptr,
        offsets=offs_a,
        mask=offs_ak[None, :] < K - pid_k * SPLITK_BLOCK_SIZE // 2,
      )
      a_scales = gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=offs_as,
        mask=offs_ks[None, :]
        < (2 * K // SCALE_GROUP_SIZE) - pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE),
      )

    if EVEN_K:
      b = gl.amd.cdna4.buffer_load(
        ptr=b_ptr,
        offsets=offs_b,
        cache=cache_modifier,
      )
      b_scales = gl.amd.cdna4.buffer_load(
        ptr=b_scales_ptr, offsets=offs_bs, cache=cache_modifier
      )
    else:
      b = gl.amd.cdna4.buffer_load(
        ptr=b_ptr,
        offsets=offs_b,
        mask=offs_bk[:, None] < K - pid_k * SPLITK_BLOCK_SIZE // 2,
        cache=cache_modifier,
      )
      b_scales = gl.amd.cdna4.buffer_load(
        ptr=b_scales_ptr,
        offsets=offs_bs,
        mask=offs_ks[None, :]
        < (2 * K // SCALE_GROUP_SIZE) - pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE),
        cache=cache_modifier,
      )

    smem_as.store(a_scales)
    smem_a.store(a)

    accumulator = gl.zeros(
      (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
    )

    # num_stages:2
    for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter - 1):
      # advance pointers
      a_ptr += (BLOCK_SIZE_K // 2) * stride_ak
      b_ptr += (BLOCK_SIZE_K // 2) * stride_bk
      a_scales_ptr += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
      b_scales_ptr += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

      if EVEN_K:
        a = gl.amd.cdna4.buffer_load(
          ptr=a_ptr,
          offsets=offs_a,
        )
      else:
        a = gl.amd.cdna4.buffer_load(
          ptr=a_ptr,
          offsets=offs_a,
          mask=offs_ak[None, :] < K - (k + 1) * (BLOCK_SIZE_K // 2),
        )
      smem_b.store(b)
      smem_bs.store(b_scales)
      curr_a = smem_a.load(layout=dot_a_layout)
      curr_a_scales = smem_as.load(layout=linear_as)

      if EVEN_K:
        a_scales = gl.amd.cdna4.buffer_load(
          ptr=a_scales_ptr,
          offsets=offs_as,
        )
      else:
        a_scales = gl.amd.cdna4.buffer_load(
          ptr=a_scales_ptr,
          offsets=offs_as,
          mask=offs_ks[None, :]
          < (2 * K // SCALE_GROUP_SIZE) - (k + 1) * (BLOCK_SIZE_K // SCALE_GROUP_SIZE),
        )

      curr_b_scales = smem_bs.load(layout=linear_bs)
      if EVEN_K:
        b = gl.amd.cdna4.buffer_load(
          ptr=b_ptr,
          offsets=offs_b,
          cache=cache_modifier,
        )
        b_scales = gl.amd.cdna4.buffer_load(
          ptr=b_scales_ptr, offsets=offs_bs, cache=cache_modifier
        )
      else:
        b = gl.amd.cdna4.buffer_load(
          ptr=b_ptr,
          offsets=offs_b,
          mask=offs_bk[:, None] < K - (k + 1) * (BLOCK_SIZE_K // 2),
          cache=cache_modifier,
        )
        b_scales = gl.amd.cdna4.buffer_load(
          ptr=b_scales_ptr,
          offsets=offs_bs,
          mask=offs_ks[None, :]
          < (2 * K // SCALE_GROUP_SIZE) - (k + 1) * (BLOCK_SIZE_K // SCALE_GROUP_SIZE),
          cache=cache_modifier,
        )
      curr_b = smem_b.load(layout=dot_b_layout)

      accumulator = gl.amd.cdna4.mfma_scaled(
        a=curr_a,
        a_scale=curr_a_scales,
        a_format="e2m1",
        b=curr_b,
        b_scale=curr_b_scales,
        b_format="e2m1",
        acc=accumulator,
      )

      smem_a.store(a)
      smem_as.store(a_scales)

    # ======= Epilogue ========
    smem_b.store(b)
    smem_bs.store(b_scales)
    curr_a = smem_a.load(layout=dot_a_layout)
    curr_b = smem_b.load(layout=dot_b_layout)
    curr_a_scales = smem_as.load(layout=linear_as)
    curr_b_scales = smem_bs.load(layout=linear_bs)

    accumulator = gl.amd.cdna4.mfma_scaled(
      a=curr_a,
      a_scale=curr_a_scales,
      a_format="e2m1",
      b=curr_b,
      b_scale=curr_b_scales,
      b_format="e2m1",
      acc=accumulator,
    )

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
      0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, linear_mn)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
      0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, linear_mn)
    )
    offs_c = (
      stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    c = gl.convert_layout(c, layout=linear_mn, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c, c_ptr, offs_c, c_mask)


@gluon.jit
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
  BLOCK_SIZE_M: gl.constexpr,
  BLOCK_SIZE_N: gl.constexpr,
  ACTUAL_KSPLIT: gl.constexpr,
  MAX_KSPLIT: gl.constexpr,
):

  pid_m = gl.program_id(axis=0)
  pid_n = gl.program_id(axis=1)

  blocked_kmn: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 1, 4],
    threads_per_warp=[2, 2, 16],
    warps_per_cta=[1, 4, 1],
    order=[2, 0, 1],
  )

  blocked_mn: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 4],
    threads_per_warp=[4, 16],
    warps_per_cta=[4, 1],
    order=[1, 0],
  )

  offs_m = (
    pid_m * BLOCK_SIZE_M
    + gl.arange(
      0, BLOCK_SIZE_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_kmn))
    )
  ) % M
  offs_n = (
    pid_n * BLOCK_SIZE_N
    + gl.arange(
      0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_kmn))
    )
  ) % N
  offs_k = gl.arange(
    0, MAX_KSPLIT, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_kmn))
  )
  c_in_ptrs = (
    c_in_ptr
    + (offs_k[:, None, None] * stride_c_in_k)
    + (offs_m[None, :, None] * stride_c_in_m)
    + (offs_n[None, None, :] * stride_c_in_n)
  )

  if ACTUAL_KSPLIT == MAX_KSPLIT:
    c = gl.load(c_in_ptrs)
  else:
    c = gl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
  c = gl.sum(c, axis=0)

  c = c.to(c_out_ptr.type.element_ty)
  offs_m = (
    pid_m * BLOCK_SIZE_M
    + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mn))
  ) % M
  offs_n = (
    pid_n * BLOCK_SIZE_N
    + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_mn))
  ) % N
  c_out_ptrs = (
    c_out_ptr + (offs_m[:, None] * stride_c_out_m) + (offs_n[None, :] * stride_c_out_n)
  )
  c = gl.convert_layout(c, layout=blocked_mn, assert_trivial=False)
  gl.store(c_out_ptrs, c)


@functools.lru_cache(maxsize=1024)
def _get_config(
  M: int,
  N: int,
  K: int,
):
  if not hasattr(_get_config, "_config_dict"):
    dev = get_arch()
    if dev != "gfx950":
      raise ValueError(
        "Gluon implementation is not supported on this device (requires CDNA4)."
      )
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-AFP4WFP4.json"
    with open(fpath, "r") as file:
      config = json.load(file)
    _get_config._config_dict = config

  return _get_config._config_dict["any"]

@functools.partial(jax.jit, static_argnums=4)
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
      dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
      y (Optional[jnp.ndarray]): Pre-allocated output tensor with shape (M, N).
      config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
          BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
      skip_reduce (Optional[bool]): skip reduction, y becomes (SPK, M, N) where SPK is determined by config

  Returns:
      y (jnp.ndarray): Output with shape (M, N) or (SPK, M, N).
  """

  assert is_fp4_avail(), "MXFP4 is not available on your device"

  M, K = x.shape
  N, K = w.shape

  # Transpose w
  w = w.T

  if config is None:
    config = _get_config(M, N, K)

  if config["BLOCK_SIZE_K"] >= K * 2:
    config["NUM_KSPLIT"] = 1

  assert config["NUM_KSPLIT"] >= 1
  unit_NUM_KSPLIT = config["NUM_KSPLIT"] == 1

  if not unit_NUM_KSPLIT:
    SPLITK_BLOCK_SIZE = (
      triton.cdiv((2 * triton.cdiv(K, config["NUM_KSPLIT"])), config["BLOCK_SIZE_K"])
      * config["BLOCK_SIZE_K"]
    )
  else:
    SPLITK_BLOCK_SIZE = 2 * K

  config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE

  if not unit_NUM_KSPLIT:
    if _USE_GEMM_SPLITK_BF16:
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=dtype) #, device=x.device)
    else:
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=jnp.float32)# , device=x.device)

    y_pp_stride = (y_pp.shape[1] * y_pp.shape[2], y_pp.shape[2], 1)
  else:
    y_pp = None
    y_pp_stride = None

  if y is None and (unit_NUM_KSPLIT or not skip_reduce):
    y = jnp.empty((M, N), dtype=dtype) #, device=x.device)

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
    num_warps=config["num_warps"],
    num_stages=config["num_stages"],
    metaparams=config,
  )

  if unit_NUM_KSPLIT:
    y = result
  else:
    y_pp = result
    if skip_reduce:
      return y_pp

    REDUCE_BLOCK_SIZE_M = 16
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
      input_output_aliases={0: 1},
      grid=grid_reduce,
      BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
      BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
      ACTUAL_KSPLIT=ACTUAL_KSPLIT,
      MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
    )

  return y
