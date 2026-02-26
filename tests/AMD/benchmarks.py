from collections.abc import Callable
import itertools
import sys
import time

from benchstats import qbench as qb
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


benchmark_sets: dict[str, Callable] = {}


def register_benchmark_set(f: Callable) -> Callable:
  name_parts = f.__name__.split("_", maxsplit=-1)
  assert len(name_parts) >= 3
  assert name_parts[0] == "make" and name_parts[-1] == "benchmark"
  name = name_parts[1:-1]
  benchmark_sets["_".join(name)] = f
  return f


@register_benchmark_set
def make_gemm_afp4wfp4_benchmark() -> dict[str, tuple[Callable, Callable]]:
  if not arch_info.is_fp4_avail():
    print(
      "MXFP4 not supported on this architecture (requires CDNA4). Skipping gemm_afp4wfp4 benchmarks."
    )
    return {}

  from gemm_afp4wfp4_test import generate_gemm_afp4wfp4_inputs
  from gemm_afp4wfp4_gluon import gemm_afp4wfp4 as gluon_gemm_afp4wfp4
  from gemm_afp4wfp4_triton import gemm_afp4wfp4 as triton_gemm_afp4wfp4

  def init(M, N, K, dtype, layout="TN", output=True, skip_reduce=False) -> list:
    (x, _, w_triton, _, _, x_scales_triton, w_scales_triton, _, y) = (
      generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout=layout, output=output)
    )
    return [x, w_triton, x_scales_triton, w_scales_triton, dtype, y, None, skip_reduce]

  def instantiate_config(m_n_k, dtype, layout, output, skip_reduce, is_gluon):
    return lambda: init(
      *m_n_k, dtype, layout=layout, output=output, skip_reduce=skip_reduce
    )

  def make_benchmarks(cfg):
    return {
      f"gemm_afp4wfp4({m_n_k[0]},{m_n_k[1]},{m_n_k[2]},{dtype.__name__},{layout},"
      f"{'output' if output else 'no_output'},{'skip' if skip_reduce else 'no_skip'}) {'gluon' if is_gluon else 'triton'}": (
        instantiate_config(m_n_k, dtype, layout, output, skip_reduce, is_gluon),
        gluon_gemm_afp4wfp4 if is_gluon else triton_gemm_afp4wfp4,
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
      [True, False],  # is_gluon
    )
  )
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
        [True, False],  # is_gluon
      )
    )
  )
  return ret


@triton.jit
def copy_scalar_triton(in_ptr, out_ptr):
  value = tl.load(in_ptr)
  tl.store(out_ptr, value)


@gluon.jit
def copy_scalar_gluon(in_ptr, out_ptr):
  value = gl.load(in_ptr)
  gl.store(out_ptr, value)


@register_benchmark_set
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
    # even though the kernel shouldn't be data dependent, it's noticiably slower on
    # a random data (jax.block_until_ready() is accounted for)
    return [random.randint(random.key(42), (NGigs * 1024 * 1024 * 1024,), 0, 1000000)]
    # return [jnp.arange(NGigs * 1024 * 1024 * 1024)]

  def init_vec_out() -> list:
    i = random.randint(random.key(42), (NGigs * 1024 * 1024 * 1024,), 0, 1000000)
    # i = jnp.arange(NGigs * 1024 * 1024 * 1024)
    return [i, jnp.empty_like(i)]

  # fmt: off
  return {
    "startup(1) triton": (init_scalar, lambda x: startup(x, copy_scalar_triton)),
    "startup(1) gluon": (init_scalar, lambda x: startup(x, copy_scalar_gluon)),
    "startup(1) jcall triton": (init_scalar, jax.jit(lambda x: startup(x, copy_scalar_triton))),
    "startup(1) jcall gluon": (init_scalar, jax.jit(lambda x: startup(x, copy_scalar_gluon))),
    #
    # f"startup({NGigs}G) triton": (init_scalar, lambda x: startup(x, copy_scalar_triton)),      # about the same as
    # f"startup({NGigs}G) gluon": (init_scalar, lambda x: startup(x, copy_scalar_gluon)),        # `out jcall` below
    f"startup({NGigs}G) jcall triton": (init_vec, jax.jit(lambda x: startup(x, copy_scalar_triton))),
    f"startup({NGigs}G) jcall gluon": (init_vec, jax.jit(lambda x: startup(x, copy_scalar_gluon))),
    # jitted with output
    # f"startup({NGigs}G) out triton": (init_vec_out, lambda x, y: startup_out(x, y, copy_scalar_triton)),  # about the same as
    # f"startup({NGigs}G) out gluon": (init_vec_out, lambda x, y: startup_out(x, y, copy_scalar_gluon)),    # `out jcall` below
    f"startup({NGigs}G) out jcall triton": (init_vec_out, jax.jit(lambda x, y: startup_out(x, y, copy_scalar_triton))),
    f"startup({NGigs}G) out jcall gluon": (init_vec_out, jax.jit(lambda x, y: startup_out(x, y, copy_scalar_gluon))),
    # jitted with output donation
    f"startup({NGigs}G) donate jcall triton": (init_vec_out, jax.jit(lambda x, y: startup_out(x, y, copy_scalar_triton), donate_argnums=(1,))),
    f"startup({NGigs}G) donate jcall gluon": (init_vec_out, jax.jit(lambda x, y: startup_out(x, y, copy_scalar_gluon), donate_argnums=(1,))),
  }
  # fmt: on


"""

typical output:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                            ┃ mean (means),             ┃ median (means),           ┃ min (means),              ┃
┃ Benchmark                                  ┃ [0%, 50%, 100%]           ┃ [0%, 50%, 100%]           ┃ [0%, 50%, 100%]           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ startup(1) | triton vs gluon               │ 93.17us ~ 92.76us {-0.4%} │ 88.00us ~ 87.82us {-0.2%} │ 69.35us ~ 69.36us {+0.0%} │
│                                            │ [88.39u,91.79u,104.9u] ~  │ [84.97u,87.40u,92.13u] ~  │ [67.32u,69.31u,70.93u] ~  │
│                                            │ [87.95u,91.59u,101.4u]    │ [85.62u,87.64u,91.30u]    │ [62.17u,70.15u,72.06u]    │
│                                            │ {-0.5%,-0.2%,-3.4%}       │ {+0.8%,+0.3%,-0.9%}       │ {-7.7%,+1.2%,+1.6%}       │
│                                            │           (10 vs 10)      │           (10 vs 10)      │           (10 vs 10)      │
│ startup(1) jcall | triton vs gluon         │ 57.32us ~ 57.26us {-0.1%} │ 56.63us ~ 56.75us {+0.2%} │ 45.94us ~ 46.12us {+0.4%} │
│                                            │ [55.65u,56.88u,60.45u] ~  │ [55.71u,56.63u,57.42u] ~  │ [41.50u,46.56u,47.16u] ~  │
│                                            │ [55.01u,56.60u,60.77u]    │ [55.10u,56.44u,58.43u]    │ [44.54u,46.26u,47.27u]    │
│                                            │ {-1.1%,-0.5%,+0.5%}       │ {-1.1%,-0.3%,+1.8%}       │ {+7.3%,-0.7%,+0.2%}       │
│                                            │           (10 vs 10)      │           (10 vs 10)      │           (10 vs 10)      │
│ startup(4G) jcall | triton vs gluon        │ 81.10us ~ 86.00us {+6.0%} │ 77.72us ~ 77.95us {+0.3%} │ 53.93us ~ 52.71us {-2.3%} │
│                                            │ [72.39u,79.83u,93.65u] ~  │ [66.91u,77.22u,85.37u] ~  │ [49.83u,54.49u,56.55u] ~  │
│                                            │ [70.66u,86.64u,109.3u]    │ [65.74u,78.78u,85.21u]    │ [47.52u,52.45u,57.31u]    │
│                                            │ {-2.4%,+8.5%,+16.7%}      │ {-1.7%,+2.0%,-0.2%}       │ {-4.6%,-3.7%,+1.3%}       │
│                                            │           (10 vs 10)      │           (10 vs 10)      │           (10 vs 10)      │
│ startup(4G) out jcall | triton vs gluon    │ 7.317ms ~ 7.317ms {-0.0%} │ 7.321ms ~ 7.318ms {-0.0%} │ 7.129ms ~ 7.132ms {+0.0%} │
│                                            │ [7.308m,7.316m,7.332m] ~  │ [7.310m,7.319m,7.337m] ~  │ [7.021m,7.145m,7.190m] ~  │
│                                            │ [7.300m,7.318m,7.331m]    │ [7.309m,7.318m,7.330m]    │ [7.070m,7.134m,7.185m]    │
│                                            │ {-0.1%,+0.0%,-0.0%}       │ {-0.0%,-0.0%,-0.1%}       │ {+0.7%,-0.1%,-0.1%}       │
│                                            │           (10 vs 10)      │           (10 vs 10)      │           (10 vs 10)      │
│ startup(4G) donate jcall | triton vs gluon │ 72.86us ~ 71.88us {-1.3%} │ 70.42us ~ 69.22us {-1.7%} │ 53.29us ~ 51.84us {-2.7%} │
│                                            │ [66.79u,73.47u,77.65u] ~  │ [63.56u,71.49u,76.13u] ~  │ [47.68u,53.85u,57.62u] ~  │
│                                            │ [65.06u,72.56u,76.05u]    │ [63.84u,68.57u,75.14u]    │ [48.56u,51.28u,55.61u]    │
│                                            │ {-2.6%,-1.2%,-2.1%}       │ {+0.4%,-4.1%,-1.3%}       │ {+1.8%,-4.8%,-3.5%}       │
│                                            │           (10 vs 10)      │           (10 vs 10)      │           (10 vs 10)      │
└────────────────────────────────────────────┴───────────────────────────┴───────────────────────────┴───────────────────────────┘
"""


def names_to_comparisons(names: list | tuple) -> tuple[list[str], str]:
  res = []
  for n in names:
    assert isinstance(n, str)
    common, alt = n.rsplit(sep=" ", maxsplit=1)
    res.append(f"{common}|{alt}")
  return res, "|"


def get_benchmark_results(
  bms: dict[str, tuple[Callable, Callable]],
  iters: int = 100,
  reps: int = 10,
  warmup: int = 3,
  show_progress: bool = True,
) -> np.ndarray:
  n_bms = len(bms)
  assert n_bms > 0

  if show_progress:
    progress = Progress(transient=True)
    warmup_task = progress.add_task("Warmup", total=warmup)
    iter_task = progress.add_task("Iterations", total=iters)
    reps_task = progress.add_task("Repetitions", total=reps)
    rwarmup = progress.track(range(warmup), task_id=warmup_task)
    rreps = progress.track(range(reps), task_id=reps_task)
    progress.start()
  else:
    rreps = range(reps)
    rwarmup = range(warmup)

  for w in rwarmup:
    for i, f in bms.values():
      f(*i())

  if show_progress:
    progress.update(warmup_task, visible=False)

  results = np.empty((n_bms, reps, iters), dtype=np.float64)
  bm_idxs = np.arange(n_bms, dtype=np.int32)
  rng = np.random.default_rng()
  funcs = tuple(bms.values())  # dictionaries are ordered, the order is stable

  for r in rreps:
    if show_progress:
      progress.update(iter_task, completed=0)
    for i in range(iters):
      # reshuffling the order of bm invocation to break HW dependencies
      rng.shuffle(bm_idxs)
      for bm_idx in bm_idxs:
        init_func, bm_func = funcs[bm_idx]
        inputs = init_func()
        for idx, inp in enumerate(inputs):
          inputs[idx] = jax.block_until_ready(inp)

        start = time.perf_counter_ns()
        o = jax.block_until_ready(bm_func(*inputs))
        end = time.perf_counter_ns()

        results[bm_idx, r, i] = (end - start) * 1e-9

      if show_progress:
        progress.update(iter_task, completed=i)

  if show_progress:
    progress.stop()

  if 1 == n_bms:
    results = results[0]
  return results


def main(enabled=[]):
  # jax.config.update("jax_enable_x64", True)
  start = time.perf_counter_ns()

  if not enabled:
    enabled = benchmark_sets.keys()

  if len(frozenset(enabled)) != len(enabled):
    raise ValueError("Benchmark set names must be unique")

  bms = {}
  for bm_id in enabled:
    if bm_id not in benchmark_sets:
      raise ValueError(
        f"Benchmark {bm_id} not found, available benchmark sets are: {', '.join(benchmark_sets.keys())}"
      )

    b = benchmark_sets[bm_id]()
    assert frozenset(b.keys()) not in bms, "Some benchmark ids are colliding!"
    bms.update(b)

  assert len(bms) > 0, "No benchmark sets were enabled, shouildn't be here"

  bm_names, alt_delimiter = names_to_comparisons(tuple(bms.keys()))
  all_bms = '", "'.join(bm_names)
  print(f'Going run {len(bm_names)} benchmarks: "{all_bms}"')

  results = get_benchmark_results(bms, iters=100, reps=10)
  qb.showBench(
    results,
    bm_names,
    alt_delimiter,
    metrics={"mean": np.mean, "median": np.median, "min": np.min},
  )

  end = time.perf_counter_ns()
  print(f"Done in {makeReadable((end - start) * 1e-9, 1)}s")


if __name__ == "__main__":
  main(sys.argv[1:])
