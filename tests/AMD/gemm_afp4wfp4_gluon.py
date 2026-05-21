# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b/aiter/ops/triton/gluon/gemm_afp4wfp4.py
# Kernels are copypasted, the launch function was rewritten to use JAX primitives and
# match jax-triton calling conventions

import functools
import json
from typing import Optional, Tuple

from strided_array import StridedArray

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import _aiter
import arch_info

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


@triton.heuristics({
  "EVEN_K": lambda args: (
    (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
    and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
    and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0)
  ),
})
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
  pid_unified = _aiter.remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

  pid_k = pid_unified % NUM_KSPLIT
  pid = pid_unified // NUM_KSPLIT
  num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

  if NUM_KSPLIT == 1:
    pid_m, pid_n = _aiter.pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
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
    dev = arch_info.get_arch()
    if dev != "gfx950":
      raise ValueError(
        "Gluon implementation is not supported on this device (requires CDNA4)."
      )
    fpath = f"{_aiter.AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-AFP4WFP4.json"
    with open(fpath, "r") as file:
      config = json.load(file)
    _get_config._config_dict = config

  return _get_config._config_dict["any"]


def gemm_afp4wfp4(
  x: StridedArray,
  w: StridedArray,
  x_scales: StridedArray,
  w_scales: StridedArray,
  dtype: Optional[jnp.dtype] = jnp.bfloat16,
  y: Optional[jnp.ndarray] = None,
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
      y (if provided)  : jnp.ndarray of shape (M, N), C-contiguous

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
      y: Optional pre-allocated output of shape (M, N).
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

  assert x.strides == (K2, 1), (
    f"x.strides={x.strides}; expected ({K2}, 1)"
  )
  assert w.strides == (K2, 1), (
    f"w.strides={w.strides}; expected ({K2}, 1) (pre-T)"
  )
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
    y,
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
  donate_argnames="y",
)
def _gemm_afp4wfp4_jit(
  x_data: jnp.ndarray,
  w_data: jnp.ndarray,
  x_scales_data: jnp.ndarray,
  w_scales_data: jnp.ndarray,
  y: Optional[jnp.ndarray],
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
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=dtype)
    else:
      y_pp = jnp.empty((config["NUM_KSPLIT"], M, N), dtype=jnp.float32)

    y_pp_stride = (y_pp.shape[1] * y_pp.shape[2], y_pp.shape[2], 1)
  else:
    y_pp = None
    y_pp_stride = None

  if y is None and (unit_NUM_KSPLIT or not skip_reduce):
    y = jnp.empty((M, N), dtype=dtype)

  grid = lambda META: (  # noqa: E731
    (
      META["NUM_KSPLIT"]
      * triton.cdiv(M, META["BLOCK_SIZE_M"])
      * triton.cdiv(N, META["BLOCK_SIZE_N"])
    ),
  )

  out_tensor_y = y if unit_NUM_KSPLIT else y_pp

  result = jt.triton_call(
    x_data,
    w_data,
    out_tensor_y,
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
    y.shape[1] if unit_NUM_KSPLIT else y_pp_stride[1],
    1 if unit_NUM_KSPLIT else y_pp_stride[2],
    x_scales_strides[0],
    x_scales_strides[1],
    w_scales_strides[0],
    w_scales_strides[1],
    kernel=_gemm_afp4wfp4_kernel,
    input_output_aliases={2: 0},
    out_shape=jax.ShapeDtypeStruct(shape=out_tensor_y.shape, dtype=out_tensor_y.dtype),
    grid=grid,
    **config,
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
      input_output_aliases={1: 0},
      grid=grid_reduce,
      BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
      BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
      ACTUAL_KSPLIT=ACTUAL_KSPLIT,
      MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
    )

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
