"""supplemental code from aiter to alleviate porting kernel tests."""
# based on https://github.com/ROCm/aiter/blob/7411c99753f0661a3eecdbdb1b36feb58539f62b

import json
import functools
import os

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

import arch_info


AITER_TRITON_CONFIGS_PATH: str | None = None

if AITER_TRITON_CONFIGS_PATH is None:
  AITER_TRITON_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "aiter_configs")


# Standard bounds for M_LEQ_x keys (tuple for hashability with LRU cache)
STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

# This flag should be set to True, unless it is being used for debugging
USE_LRU_CACHE = True


def serialize_dict(d: dict) -> str:
  return json.dumps(d)


def deserialize_str(s: str) -> dict:
  return json.loads(s)


def _load_config_file(
  cache_dict: dict,
  cache_key: str,
  fpath: str,
  config_key: str,
  fpath_should_exist: bool = False,
) -> bool:
  """
  Helper function to load a config file and cache it.
  """
  if os.path.exists(fpath):
    with open(fpath, "r") as file:
      config = json.load(file)
    cache_dict[cache_key][config_key] = config
    return True
  elif fpath_should_exist:
    raise AssertionError(f"Required config file doesn't exist: {fpath}")
  return False


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_gemm_config(
  config_name: str,
  M: int,
  N: int | None = None,
  K: int | None = None,
  bounds: tuple[int, ...] | None = None,
  specialized_filename: str | None = None,
) -> tuple[dict, bool]:
  """
  Load a GEMM configuration using the standardized M_LEQ_x/M_GEQ_y/any format.

  This function provides a unified way to load GEMM configs across all kernels.
  It uses the following logic:
  1. Load default config file: {arch}-{config_name}.json
  2. If N and K are provided, try to load specialized config: {arch}-{config_name}-N={N}-K={K}.json
     Or if specialized_filename is provided, use: {arch}-{config_name}-{specialized_filename}.json
  3. Search for M_LEQ_x keys in order of bounds (default: STANDARD_M_BOUNDS)
  4. If no M_LEQ_x matches, search for M_GEQ_x keys in reverse order
  5. Fall back to "any" if no bounds match

  Args:
      config_name: Name of the config (example - "GEMM-A16W16")
      M: M dimension of the GEMM
      N: N dimension of the GEMM (optional)
      K: K dimension of the GEMM (optional)
      bounds: Custom bounds to use instead of STANDARD_M_BOUNDS (optional)
      specialized_filename: Custom specialized filename suffix (optional)

  Returns:
      Dictionary with the config params,
      bool indicating if the config is tuned.(True if tuned, False otherwise)
  """
  # Input validation
  assert M >= 0, "M must be positive."
  assert N is None or N > 0, "N must be positive when provided."
  assert K is None or K > 0, "K must be positive when provided."
  assert bounds is None or (
    len(bounds) > 0
    and all(x > 0 for x in bounds)
    and all(x < y for x, y in zip(bounds, bounds[1:]))
  ), (
    "When provided, bounds must be a non-empty tuple of strictly increasing positive numbers."
  )

  if not hasattr(get_gemm_config, "_config_cache"):
    get_gemm_config._config_cache = {}

  dev = arch_info.get_arch()
  cache_key = f"{dev}_{config_name}"

  if cache_key not in get_gemm_config._config_cache:
    get_gemm_config._config_cache[cache_key] = {}

    # Load default config (must exist)
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}.json"
    _load_config_file(
      get_gemm_config._config_cache,
      cache_key,
      fpath,
      "default",
      fpath_should_exist=True,
    )

  config_dict_key = "default"

  # Handle custom specialized filename (for fused kernels with multiple N dims)
  if specialized_filename is not None:
    spec_key = specialized_filename
    if spec_key not in get_gemm_config._config_cache[cache_key]:
      fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}-{specialized_filename}.json"
      if _load_config_file(get_gemm_config._config_cache, cache_key, fpath, spec_key):
        config_dict_key = spec_key
    else:
      config_dict_key = spec_key

  elif N is not None and K is not None:
    nk_key = f"{N}_{K}"
    if nk_key not in get_gemm_config._config_cache[cache_key]:
      # load specialized config
      fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}-N={N}-K={K}.json"
      if _load_config_file(get_gemm_config._config_cache, cache_key, fpath, nk_key):
        config_dict_key = nk_key
    else:
      config_dict_key = nk_key

  config_dict = get_gemm_config._config_cache[cache_key][config_dict_key]

  # use standard bounds unless custom bounds are passed
  search_bounds = bounds if bounds is not None else STANDARD_M_BOUNDS

  # Search for M_LEQ_x keys
  for bound in search_bounds:
    key = f"M_LEQ_{bound}"
    if M <= bound and key in config_dict:
      return dict(config_dict[key]), config_dict_key != "default"

  # Search for M_GEQ_x keys
  for bound in reversed(search_bounds):
    key = f"M_GEQ_{bound}"
    if M >= bound and key in config_dict:
      return dict(config_dict[key]), config_dict_key != "default"

  if "any" in config_dict:
    return dict(config_dict["any"]), False

  raise KeyError(
    f"No matching configuration found for M={M}, N={N}, K={K} in config '{config_name}'."
  )


#######################################################################################


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
  ## pid remapping on xcds
  # Number of pids per XCD in the new arrangement
  pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
  # When GRID_MN cannot divide NUM_XCDS, some xcds will have
  # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
  # We calculate the number of xcds that have pids_per_xcd pids as
  # tall_xcds
  tall_xcds = GRID_MN % NUM_XCDS
  tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
  # Compute current XCD and local pid within the XCD
  xcd = pid % NUM_XCDS
  local_pid = pid // NUM_XCDS
  # Calculate new pid based on the new grouping
  # Note that we need to consider the following two cases:
  # 1. the current pid is on a tall xcd
  # 2. the current pid is on a short xcd
  if xcd < tall_xcds:
    pid = xcd * pids_per_xcd + local_pid
  else:
    pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

  return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
  """
  Maps 1D pid to 2D grid coords (pid_m, pid_n).

  Args:
      - pid: 1D pid
      - num_pid_m: grid m size
      - num_pid_n: grid n size
      - GROUP_SIZE_M: tl.constexpr: default is 1
  """
  if GROUP_SIZE_M == 1:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
  else:
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    tl.assume(group_size_m >= 0)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

  return pid_m, pid_n


#######################################################################################

# dtypes
defaultDtypes = {
  "gfx942": {"fp8": jnp.float8_e4m3fnuz},
  "gfx950": {"fp8": jnp.float8_e4m3fn},
  "gfx1250": {"fp8": jnp.float8_e4m3fn},
}

_8bit_fallback = jnp.uint8


def get_dtype_fp8():
  return defaultDtypes.get(arch_info.get_arch(), {"fp8": _8bit_fallback})["fp8"]


i4x2 = getattr(jnp, "int4", _8bit_fallback)
fp4x2 = getattr(jnp, "float4_e2m1fn_x2", _8bit_fallback)
fp8 = get_dtype_fp8()
fp8_e8m0 = getattr(jnp, "float8_e8m0fnu", _8bit_fallback)
fp16 = jnp.float16
bf16 = jnp.bfloat16
fp32 = jnp.float32
u32 = jnp.uint32
i32 = jnp.int32
i16 = jnp.int16
i8 = jnp.int8

d_dtypes = {
  "fp8": fp8,
  "fp8_e8m0": fp8_e8m0,
  "fp16": fp16,
  "bf16": bf16,
  "fp32": fp32,
  "i4x2": i4x2,
  "fp4x2": fp4x2,
  "u32": u32,
  "i32": i32,
  "i16": i16,
  "i8": i8,
}


#######################################################################################


@functools.lru_cache()
def get_dtype_max(dtype):
  if jnp.issubdtype(dtype, jnp.floating):
    dtypeMax = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtypeMax = jnp.iinfo(dtype).max
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")
  return dtypeMax


def pertoken_quant(
  x,
  scale=None,
  x_scale=None,  # smooth_scale
  scale_dtype=fp32,
  quant_dtype=i8,
  dtypeMax=None,
):
  x = x.astype(fp32)
  if x_scale is None:
    hidden_states = x
  else:
    # smooth quant
    hidden_states = x * x_scale

  if dtypeMax is None:
    dtypeMax = get_dtype_max(quant_dtype)

  per_token_scale = scale
  if scale is None:
    # [m, 1]
    # per_token_amax, _ = torch.max(input=torch.abs(hidden_states), dim=-1, keepdim=True)
    per_token_amax, _ = jnp.max(jnp.abs(hidden_states), axis=-1, keepdims=True)
    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1

  # quant hidden_states
  y = (hidden_states / per_token_scale).astype(quant_dtype)
  y_scale = per_token_scale.astype(scale_dtype)
  return y, y_scale

def per_tensor_quant(
    x, scale=None, scale_dtype=fp32, quant_dtype=i8, dtypeMax=None
):
    x = x.astype(fp32)
    if scale is None:
        if dtypeMax is None:
            dtypeMax = get_dtype_max(quant_dtype)
        scale = jnp.abs(x).max() / dtypeMax
    y = x / scale

    scale = jnp.reshape(scale,(-1,))
    assert scale.size == 1

    return y.astype(quant_dtype), scale.astype(scale_dtype)
