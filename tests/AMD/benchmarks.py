import argparse
from functools import partial
from collections.abc import Callable
import itertools
import os
import time

from benchstats import qbench as qb
from benchstats.common import LoggingConsole
from benchstats.render import makeReadable
import jax
import jax.numpy as jnp
import jax.random as random
import jax_triton as jt
import numpy as np
from rich.progress import Progress
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import arch_info


######################################################################  gemm_afp4wfp4


@qb.registerBenchmark
def make_gemm_afp4wfp4_benchmark() -> dict[str, tuple[Callable, Callable]]:
  if not arch_info.is_fp4_avail():
    print(
      "MXFP4 not supported on this architecture (requires CDNA4). Skipping gemm_afp4wfp4 benchmarks."
    )
    return {}

  from gemm_afp4wfp4_test import generate_gemm_afp4wfp4_inputs
  from gemm_afp4wfp4_gluon import gemm_afp4wfp4 as gluon_gemm_afp4wfp4
  from gemm_afp4wfp4_triton import gemm_afp4wfp4 as triton_gemm_afp4wfp4

  k = jax.random.key(42)

  def init(M, N, K, dtype, layout="TN", output=True, skip_reduce=False) -> list:
    nonlocal k
    if _random_inputs:
      k, s = jax.random.split(k)
    else:
      s = k
    (x, _, w_triton, _, _, x_scales_triton, w_scales_triton, _, y) = (
      generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout=layout, output=output, key=s)
    )
    return [x, w_triton, x_scales_triton, w_scales_triton, dtype, y, None, skip_reduce]

  def instantiate_config(m_n_k, dtype, layout, output, skip_reduce, is_gluon):
    return lambda: init(
      *m_n_k, dtype, layout=layout, output=output, skip_reduce=skip_reduce
    )

  def make_benchmarks(cfg):
    return {
      f"gemm_afp4wfp4({m_n_k[0]},{m_n_k[1]},{m_n_k[2]},{dtype.__name__},{layout},"
      f"{'output' if output else 'no_output'},{'skip' if skip_reduce else 'no_skip'})|{'gluon' if is_gluon else 'triton'}": (
        gluon_gemm_afp4wfp4 if is_gluon else triton_gemm_afp4wfp4,
        instantiate_config(m_n_k, dtype, layout, output, skip_reduce, is_gluon),
      )
      for m_n_k, dtype, layout, output, skip_reduce, is_gluon in cfg
    }

  # gluon kernel config is optimized for M=N=K=8192, so to compare apples to apples,
  # trying values around that
  drop_similar = True  # don't benchmark configs yielding not much different results

  ret = make_benchmarks(
    itertools.product(
      [
        (8192, 8192, 8192),
      ],
      [jnp.bfloat16] if drop_similar else [jnp.bfloat16, jnp.float16],
      ["TN"] if drop_similar else ["TN", "TT", "NN", "NT"],
      [False] if drop_similar else [True, False],
      [False] if drop_similar else [True, False],
      [False, True],  # is_gluon
    )
  )
  if not _single_gemm:
    ret.update(
      make_benchmarks(
        itertools.product(
          [
            (4 * 8192, 4 * 8192, 4 * 8192),
            (8192 // 2, 8192 // 2, 8192 // 2),
            (8192 // 4, 8192 // 4, 8192 // 4),
            (4 * 8192, 8192, 4 * 8192),
            (8192, 4 * 8192, 8192),
            (8192 // 2, 8192 * 2, 8192 // 2),
            (8192 // 4, 8192 * 4, 8192 // 4),
            (8192 * 4, 8192 // 4, 8192 * 4),
            (4864, 8192, 4160),
            (16, 16384, 3328 * 2),
            # (128, 16384, 3328 * 2),
            (256, 256, 256),
            (256, 8192, 256),
            (1, 1, 32),
          ],
          [jnp.bfloat16],  # [jnp.bfloat16, jnp.float16],
          ["TN"],  # ["TN", "TT", "NN", "NT"],
          [False],  # [True, False],
          [False],  # [True, False],
          [False, True],  # is_gluon
        )
      )
    )
  return ret


######################################################################  startup


@triton.jit
def copy_scalar_triton(in_ptr, out_ptr):
  value = tl.load(in_ptr)
  tl.store(out_ptr, value)


@gluon.jit
def copy_scalar_gluon(in_ptr, out_ptr):
  value = gl.load(in_ptr)
  gl.store(out_ptr, value)


# @register_benchmark_set
@qb.registerBenchmark
def make_startup_benchmark() -> dict[str, tuple[Callable, Callable]]:
  """
  Prepares benchmarking startup costs of triton vs gluon kernels.

  Returns a dictionary of functions func_name->(init_func, bm_func), where bm_func takes
  a list of arguments created by init_func and returns the operation result. The runner
  takes care of doing jax.block_until_ready() on the input arguments and the result for
  benchmarking. Runtime of bm_func is measured.

  Types of benchmark we want to see:
  non-jitted:
  - startup costs - how much does it cost to just launch a trivial kernel?
      - not much, dozens microseconds
  - does cost change if we supply large arrays?
      - there's an additional cost of transferring the array to the GPU memory, but the
        cost of invocation is about the same.

  jitted:
  - does jitting the launcher changes the startup costs?
    - makes the invocation a few dozen percents faster.
  - does the cost of jitted launch change if we supply large arrays?
    - gets about the same speedup as for a scalar argument, which is basically hidden by
      the cost of transferring arrays to/from the GPU memory.
    - most important caveat: using an input-output argument without donating it to the
        launcher has an additional cost of copying that argument before
        passing it to the kernel (JAX arrays are immutable by default). This *might* not
        happen if the in-out argument is never used later (or specifically assigned to
        the result of the kernel invocation? - this wasn't tested) AND the whole
        launcher invocation is jitted so a compiler could see that the argument is never
        used later, but this might be a fragile assumption. Beware.

  jitted with donation:
  - does the cost of jitted launch change if we supply args with donation?
    - removes the additional cost of copying the argument before passing it to the
      kernel.
  """
  NGigs = 4

  def startup(input: jnp.ndarray, kernel) -> jnp.ndarray:
    return jt.triton_call(
      input,
      kernel=kernel,
      out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
      grid=1,
    )

  # beware - that launcher better be called with jitting and donation of the output
  def startup_out(input: jnp.ndarray, output: jnp.ndarray, kernel) -> jnp.ndarray:
    return jt.triton_call(
      input,
      output,
      kernel=kernel,
      out_shape=jax.ShapeDtypeStruct(shape=input.shape, dtype=input.dtype),
      grid=1,
      input_output_aliases={1: 0},
    )

  def init_scalar() -> list:
    return [jnp.array(42.314)]

  def init_vec() -> list:
    return [jnp.arange(NGigs * 1024 * 1024 * 1024)]

  def init_vec_out() -> list:
    i = jnp.arange(NGigs * 1024 * 1024 * 1024)
    return [i, jnp.empty_like(i)]

  # fmt: off
  return {
    "startup(1)|triton": (lambda x: startup(x, copy_scalar_triton), init_scalar),
    "startup(1)|gluon": (lambda x: startup(x, copy_scalar_gluon), init_scalar),
    "startup(1) jcall|triton": (jax.jit(lambda x: startup(x, copy_scalar_triton)), init_scalar),
    "startup(1) jcall|gluon": (jax.jit(lambda x: startup(x, copy_scalar_gluon)), init_scalar),
    #
    # f"startup({NGigs}G)|triton": (lambda x: startup(x, copy_scalar_triton), init_scalar),      # about the same as
    # f"startup({NGigs}G)|gluon": (lambda x: startup(x, copy_scalar_gluon), init_scalar),        # `out jcall` below
    f"startup({NGigs}G) jcall|triton": (jax.jit(lambda x: startup(x, copy_scalar_triton)), init_vec),
    f"startup({NGigs}G) jcall|gluon": (jax.jit(lambda x: startup(x, copy_scalar_gluon)), init_vec),
    # jitted with output
    # f"startup({NGigs}G) out|triton": (lambda x, y: startup_out(x, y, copy_scalar_triton), init_vec_out),  # about the same as
    # f"startup({NGigs}G) out|gluon": (lambda x, y: startup_out(x, y, copy_scalar_gluon), init_vec_out),    # `out jcall` below
    f"startup({NGigs}G) out jcall|triton": (jax.jit(lambda x, y: startup_out(x, y, copy_scalar_triton)), init_vec_out),
    f"startup({NGigs}G) out jcall|gluon": (jax.jit(lambda x, y: startup_out(x, y, copy_scalar_gluon)), init_vec_out),
    # jitted with output donation
    f"startup({NGigs}G) donate jcall|triton": (jax.jit(lambda x, y: startup_out(x, y, copy_scalar_triton), donate_argnums=(1,)), init_vec_out),
    f"startup({NGigs}G) donate jcall|gluon": (jax.jit(lambda x, y: startup_out(x, y, copy_scalar_gluon), donate_argnums=(1,)), init_vec_out),
  }
  # fmt: on


######################################################################  runner stuff


class CacheFlusher:
  def __init__(self, _cache_size_bytes=256 * 1024 * 1024):
    _clearers = {}
    for d in jax.devices():
      _clearers[d] = jnp.arange(_cache_size_bytes // 4, dtype=jnp.uint32, device=d)
    self._clearers = _clearers

    # impl in Triton just zeroes a tensor. Read-write might be a bit more reliable
    @triton.jit
    def _inc_kernel(output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < N
      values = tl.load(output_ptr + offsets, mask=mask)
      tl.store(output_ptr + offsets, values + 1, mask=mask)

    @partial(jax.jit, donate_argnums=(0,))
    def _reset_array(arr: jnp.ndarray):
      assert arr.dtype == jnp.uint32
      N = arr.size
      BLOCK_SIZE = 1024
      grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
      return jt.triton_call(
        arr,
        kernel=_inc_kernel,
        out_shape=arr,
        grid=grid,
        input_output_aliases={0: 0},
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
      )

    self._reset_array = _reset_array

  def flush(self, inputs):
    device_set = set()
    objs = []
    for inp in inputs:
      if isinstance(inp, jnp.ndarray) and inp.device not in device_set:
        device_set.add(inp.device)
        cl = self._clearers[inp.device]
        o = self._reset_array(cl)
        del cl
        self._clearers[inp.device] = o
        objs.append(o)
    for o in objs:
      jax.block_until_ready(o)


def run_benchmarks(
  enabled: list[str] | None = None,
  *,
  iters: int = 100,
  reps: int = 10,
  warmup: int = 3,
  single_gemm: bool = False,
  random_inputs: bool = True,
  randomize_iterations: bool = True,
  export_path_pfx: str | None = None,
  clear_cache: bool = True,
  batch_functions: bool = False,
):
  global _single_gemm, _random_inputs
  _single_gemm = single_gemm
  _random_inputs = random_inputs

  start = time.perf_counter_ns()

  if not enabled:
    enabled = qb.getRegisteredBenchmarkSetNames()

  if clear_cache:
    print("Will flush L2 cache before each run")
    cache_flusher = CacheFlusher()

    def clear_l2(inputs):
      cache_flusher.flush(inputs)

  else:
    print("L2 cache WON'T be flushed!")

    def clear_l2(inputs):
      pass

  if export_path_pfx:
    enabled_str = "-".join(enabled)
    fname_base = (
      f"{export_path_pfx}_{enabled_str}"
      f"_i{iters}_r{reps}"
      + ("_SG" if single_gemm else "")
      + ("" if random_inputs else "_NrI")
      + ("" if randomize_iterations else "_NrIT")
      + ("" if clear_cache else "_NCC")
      + "_"
    )
    c = 0
    while True:
      f = f"{fname_base}{c}.svg"
      if os.path.exists(f):
        c += 1
      else:
        fname_base = f
        break

    console = LoggingConsole(record=True)
  else:
    fname_base = None
    console = LoggingConsole()

  bms = qb.getRegisteredBenchmarks(enabled)
  assert len(bms) > 0, "No benchmark sets were enabled, shouildn't be here"

  bm_names = tuple(bms.keys())
  all_bms = '", "'.join(bm_names)
  print(
    f'Going to run {len(bm_names)} benchmarks ({jax.local_device_count()} device(s) available): "{all_bms}"'
  )
  if jax.local_device_count() > 1:
    console.warning(
      "More than 1 device is visible. For potentially better results consistency, restrict the number of devices using HIP_VISIBLE_DEVICES or similar environment variable."
    )

  _, results = qb.benchmark(
    bms.values(),
    iters=iters,
    reps=reps,
    warmup=warmup,
    randomize_iterations=randomize_iterations,
    batch_functions=batch_functions,
    wait_complete=jax.block_until_ready,
    clear_cache=clear_l2,
    show_progress_each=5,
    bm_names=bm_names,
    alt_delimiter="|",
    metrics={"mean": np.mean, "median": np.median, "min": np.min},
    console=console,
  )

  end = time.perf_counter_ns()
  console.print(f"Done in {makeReadable((end - start) * 1e-9, 1)}s")

  if fname_base is not None:
    console.save_svg(fname_base, title=_g_ProgramName, clear=False)

    fname_base = f"{fname_base[:-4]}.txt"
    console.save_text(fname_base)

    fname_base = f"{fname_base[:-4]}.npy"
    np.save(fname_base, results)

    print(f"Saved results to {fname_base[:-4]}")

  return results


_g_ProgramName = "Triton/Gluon benchmarks runner"


def main():
  parser = argparse.ArgumentParser(
    description=_g_ProgramName,
  )
  parser = qb.makeArgumentParser(parser)

  parser.add_argument(
    "--single_gemm",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only benchmark the base gemm size (default: False)",
  )
  parser.add_argument(
    "--random_inputs",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use random input data vs deterministic arange (default: True)",
  )
  parser.add_argument(
    "--clear_cache",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Clear L2 cache before each run (default: True)",
  )

  args = parser.parse_args()
  run_benchmarks(
    enabled=args.benchmark_sets or None,
    iters=args.iters,
    reps=args.reps,
    warmup=args.warmup,
    batch_functions=args.batch_functions,
    randomize_iterations=args.randomize_iterations,
    export_path_pfx=args.export_path_pfx,
    single_gemm=args.single_gemm,
    random_inputs=args.random_inputs,
    clear_cache=args.clear_cache,
  )


if __name__ == "__main__":
  main()
