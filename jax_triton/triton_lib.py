# Copyright 2026 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for calling Triton or Triton.Gluon kernels from JAX."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
import dataclasses
import functools
from functools import cached_property
import inspect
import os
import pprint
import tempfile
import types
from typing import Any, Protocol, Union
import zlib

from absl import logging
import jax
from jax import tree_util
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.lib import gpu_triton as triton_kernel_call_lib
import jax.extend as jex
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np


import triton
from triton.compiler import compiler as tc
import triton.language as tl
from triton.runtime import autotuner
import triton._C.libtriton as _triton
import triton.backends.nvidia.compiler as cb
import triton.backends.amd.compiler as hb

import triton.experimental.gluon._runtime as gl_runtime
from triton.experimental.gluon import language as gl


os.environ["TRITON_CACHE_DIR"] = ""
_JAX_TRITON_DUMP_DIR = os.environ.get("JAX_TRITON_DUMP_DIR")
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# b/447434580: Exceeding this limit will cause Triton to emit a single trap
# instruction, which will cause the GPU to hang indefinitely. See
# triton/third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp;l=718
_TMEM_MAX_SIZE = 512

_JAX_TO_TRITON_TYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    jnp.dtype("float8_e4m3fn"): "fp8e4nv",
    jnp.dtype("float8_e5m2"): "fp8e5",
    jnp.dtype("float8_e4m3fnuz"): "fp8e4b8",
    jnp.dtype("float8_e5m2fnuz"): "fp8e5b16",
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    jnp.dtype("bool"): "i1",
}

Grid = Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[dict[str, Any]], Grid]]


def normalize_grid(grid: GridOrLambda, metaparams) -> tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


def get_triton_type(obj: Any) -> str:
  # if isinstance(obj, (jax.core.ShapedArray, state.AbstractRef)):
  if hasattr(obj, "dtype"):
    return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
  if isinstance(obj, (tl.constexpr, gl.constexpr)):
    obj = obj.value
  if isinstance(obj, bool):  # True == isinstance(True, int) !!!
    return "B"
  if isinstance(obj, int):
    if -(2**31) <= obj < 2**31:
      return "i32"
    elif 2**31 <= obj < 2**32:
      return "u32"
    elif -(2**63) <= obj < 2**63:
      return "i64"
    elif 2**63 <= obj < 2**64:
      return "u64"
    else:
      raise ValueError(f"integer overflow representing {obj}")
  if isinstance(obj, float):
    # Triton proper unconditionally treat all floats as fp32, so we should too.
    # If one wants fp64, they should use it explicitly with np.float64.
    return "fp32"
  
  """if isinstance(obj, np.float32):
    return "fp32"
  if isinstance(obj, np.float64):
    return "fp64"
  if isinstance(obj, np.float16):
    return "fp16"""

  if isinstance(obj, str):
    return "str"
  raise NotImplementedError(
      f"could not compute type name for {obj}: {type(obj)}"
  )


triton_kernel_call_p = jex.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True
triton_kernel_call_p.def_impl(
    functools.partial(xla.apply_primitive, triton_kernel_call_p)
)


@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
  return [
      core.ShapedArray(out_shape.shape, out_shape.dtype)
      for out_shape in out_shapes
  ]


def aval_size_bytes(aval):
  return np.dtype(aval.dtype).itemsize * aval.size


def make_gpu_target_cuda(device, compute_capability):
  return cb.GPUTarget("cuda", compute_capability, 32)


_IS_HIPBackend_PATCHED = False
def _patch_hip_backend():
  """
  This defuses a bomb planted into Triton's AMD-specific compilation path by
  https://github.com/triton-lang/triton/commit/37ff43c5efd6e1b84c00a599ba070a501181e832#diff-33c9a103282c05c9d9d213b94450ae7481b6db8c3c6d810f54f175b4735a3c72
  In short: there's an unconditional and totally unnecessary "import torch" directive crashing
  the code when torch isn't installed.

  Remove the patch once triton wheel package version is pinned to >= triton version with the fix.
  """
  global _IS_HIPBackend_PATCHED
  if _IS_HIPBackend_PATCHED:
    return
  _IS_HIPBackend_PATCHED = True

  if not hasattr(hb.HIPBackend, "is_within_2gb"):
    return
  try:
    hb.HIPBackend.is_within_2gb(1)
    # if we're here, either the torch is installed, or the code was fixed
  except ImportError:
    # redefining poisoned implementation. At this point, it's super unlikely a user
    # would update python package discovery paths before the real call to is_within_2gb() to make
    # `import torch` succeed, so we could assume there's just no torch in the redefinition.
    def fixed_is_within_2gb(arg):
      MAX_INT_32 = 2**31 - 1
      if hasattr(arg, "ptr_range"):
        return arg.ptr_range() <= MAX_INT_32
      return False

    hb.HIPBackend.is_within_2gb = fixed_is_within_2gb


def make_gpu_target_hip(device, compute_capability):
  # TODO(Arech): remove _patch_hip_backend() once Triton releases a fix
  _patch_hip_backend()

  arch = triton_kernel_call_lib.get_arch_details(device)
  arch = arch.split(":")[0]
  return hb.GPUTarget("hip", arch, 64)


@dataclasses.dataclass
class CompilationResult:
  binary: str
  name: str
  shared_mem_bytes: int
  ttgir: str | None
  llir: str | None


def compile_ttir_inplace(
    ttir,
    backend: [cb.CUDABackend | hb.HIPBackend],
    options: [cb.CUDAOptions | hb.HIPOptions],
    compute_capability,
    platform,
):
  if platform == "cuda":
    return compile_ttir_to_ptx_inplace(
        ttir,
        backend,
        options,
        compute_capability,
    )

  elif platform == "rocm":
    return compile_ttir_to_hsaco_inplace(
        ttir,
        backend,
        options,
        compute_capability,
    )
  else:
    raise ValueError("Unsupported device.")


def compile_ttir_to_ptx_inplace(
    ttir,
    cuda_backend: cb.CUDABackend,
    cuda_options: cb.CUDAOptions,
    compute_capability,
) -> CompilationResult:
  if cuda_options.debug:
    print(ttir)
  try:
    metadata = {}
    opt_ttir = cuda_backend.make_ttir(
        ttir, metadata, cuda_options, compute_capability
    )
    ttgir = cuda_backend.make_ttgir(
        opt_ttir,
        metadata,
        cuda_options,
        compute_capability,
    )
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if cuda_options.debug:
    print(ttgir)
  try:
    llir = cuda_backend.make_llir(
        ttgir,
        metadata,
        cuda_options,
        compute_capability,
    )
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTGIR->LLIR pass failed!") from e
  if metadata["tmem_size"] > _TMEM_MAX_SIZE:
    raise ValueError(
        f"TMEM size {metadata['tmem_size']} exceeds limit {_TMEM_MAX_SIZE}."
    )
  shared_mem_bytes = metadata["shared"]
  if cuda_options.debug:
    print(llir)
  ptx = cuda_backend.make_ptx(
      llir,
      metadata,
      cuda_options,
      compute_capability,
  )
  if cuda_options.debug:
    print(ptx)
  name = metadata["name"]
  ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
  llir = str(llir) if _JAX_TRITON_DUMP_DIR else None
  return CompilationResult(
      binary=ptx,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=ttgir,
      llir=llir,
  )


def compile_ttir_to_hsaco_inplace(
    ttir,
    hip_backend: hb.HIPBackend,
    hip_options: hb.HIPOptions,
    compute_capability,
) -> CompilationResult:
  if hip_options.debug:
    print(ttir)
  try:
    metadata = {}
    opt_ttir = hip_backend.make_ttir(ttir, metadata, hip_options)
    ttgir = hip_backend.make_ttgir(opt_ttir, metadata, hip_options)
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if hip_options.debug:
    print(ttgir)
  try:
    llir = hip_backend.make_llir(ttgir, metadata, hip_options)
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTGIR->LLIR pass failed!") from e
  shared_mem_bytes = metadata["shared"]
  if hip_options.debug:
    print(llir)

  amdgcn = hip_backend.make_amdgcn(llir, metadata, hip_options)
  hsaco = hip_backend.make_hsaco(amdgcn, metadata, hip_options)

  name = metadata["name"]
  ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
  llir = str(llir) if _JAX_TRITON_DUMP_DIR else None
  # Instead of passing hsaco which are "bytes", we first write
  # to a file and then pass the "string" path. This is needed because
  # nanobind doesn't automatically convert between bytes and string.
  # https://github.com/wjakob/nanobind/discussions/137
  fd, hsaco_path = tempfile.mkstemp()
  with os.fdopen(fd, "wb") as f:
    f.write(hsaco)
  return CompilationResult(
      binary=hsaco_path,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=ttgir,
      llir=llir,
  )


def make_backend(
  make_gpu_target_func, compute_capability: int | None, num_ctas: int
) -> tuple[triton.compiler.BaseBackend, triton.compiler.GPUTarget, int]:
  """Resolves compute_capability and spawns triton's Backend and GPUTarget objects"""

  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general. See also how Triton did this in JITFunction's
  # `self.device_caches = defaultdict(self.create_binder)` -- it spawns a new set of
  # precomputes for each new device with `x,y,z = self.device_caches[device]` using the
  # create_binder() factory function.
  device = 0
  if compute_capability is None:
    compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  if num_ctas > 1 and compute_capability < 90:
    raise ValueError("num_ctas > 1 unsupported before Hopper.")

  gpu_target = make_gpu_target_func(device, compute_capability)
  backend = triton.compiler.make_backend(gpu_target)

  return backend, gpu_target, compute_capability


_COMPILED_KERNEL_CACHE = {}  # TODO(cjfj): Convert to LRU cache?


#################################### FOR RESEARCH ONLY
def create_function_from_signature(sig, kparams, backend):
    """
    - sig is fn.signature, result of inspect.signature(fn)
    - kparams is fn.params, a list of KernelParam objects
    - backend is the triton backend object

    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
            # TODO warn if specialization is disallowed
            # for constexpr functions results are different from what specialize_impl() returns
            # but due two 2-phase cache key resolution and work of replace_callables(),
            # the caching ends up being correct. (but still need to test both func passing
            # variants)
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            # note that runtime specialization might generate strings for some values
            # such as callables, which, if not handled carefully, might produce wrong
            # attributes later, since `parse_attr(desc)` only checks if the string
            # description has certain characters.
            # TODO: how to "fake" callables during tracing time?
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    arg = lambda x: x[0] if x[1].default is inspect.Parameter.empty else f"{x[0]}=default_{x[0]}"
    # arg returns a func signature argument declaration reconstructed from the original func
    # signature - either a param name if it has no default, or param=default_val otherwise

    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""
    # order of the func params declaration is determined by the fn signature and then a
    # blank **options: e.g. "a_ptr, b_ptr, ..., BLOCK_SIZE_M, num_warps, waves_per_eu, ..., **options"
    #
    # params var is just a dict mapping fn param name to its value, using the sig object again, e.g.:
    # "{'a_ptr': a_ptr, ..., 'BLOCK_SIZE_M': BLOCK_SIZE_M, 'num_warps': num_warps, 'waves_per_eu': waves_per_eu, ...}"
    # We call this named_args and use it very early.
    #
    # specialization is a list constructed from either pre-created specialization strings,
    # or from the result of calling specialize_impl() for each non-constexpr param.
    # Specializations are used later to extract additional constexprs

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    specialize_impl = _triton.native_specialize_impl
    func_namespace["specialize_impl"] = specialize_impl
    func_namespace["backend"] = backend
    func_namespace["JITCallable"] = triton.runtime.jit.JITCallable

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']
#################################### END of FOR RESEARCH ONLY


_BACKEND_OPTIONS_FIELD_NAMES = {
  "cuda": frozenset(cb.CUDAOptions.__dataclass_fields__.keys()),
  "hip": frozenset(hb.HIPOptions.__dataclass_fields__.keys()),
}


def get_or_create_triton_kernel(
  make_gpu_target_func,
  platform: str,
  fn: triton.JITFunction | gl_runtime.GluonJITFunction,
  args: list[Any],
  arg_dtypes: list[str],
  scalar_args: dict[int, Any],
  *,
  # num_warps,
  # num_stages,
  # num_ctas,
  compute_capability: int | None,
  # enable_fp_fusion,
  metaparams: dict[str, Any],
  # debug: bool,
) -> tuple[triton_kernel_call_lib.TritonKernel, Any]:
  # TODO: some vars below could be removed
  num_warps = metaparams["num_warps"]
  num_stages = metaparams["num_stages"]
  num_ctas = metaparams["num_ctas"]
  enable_fp_fusion = metaparams["enable_fp_fusion"]
  assert all(isinstance(v, int) for v in (num_warps, num_stages, num_ctas))  # sanity

  backend, gpu_target, compute_capability = make_backend(
    make_gpu_target_func, compute_capability, num_ctas
  )

  # TODO(Arech): triton.runtime.jit.create_function_from_signature() is better here
  # The binder it generates takes kernel positional argument values (need to pass args here)
  # and all kwargs passed to the triton's launcher (as options, for us it's metaparams)
  # and returns a tuple of 3: 1 is a dict kernel_param_name->value mapping each
  # (positional and key-value) argument name to its value; 2 is a list of
  # specializations; 3 seems to be just forwarding of the options/kwargs/metaparams back.
  # The dynamically generated binder function exists purely for performance — it avoids
  # the overhead of branched algo of building the specialization and application of
  # defaults on every kernel launch, while still computing the specialization tuples
  # from the actual runtime argument values.
  #
  # A specialization is a list[tuple[str, Any]], one entry per kernel parameter, that
  # captures two things about each argument at call time:
  #   1. Element 0 — the type string: e.g. "i32", "i64", "*fp16", "*ki32"
  #     (pointer to const), "u1", "fp32", "constexpr", "tensordesc<...>".
  #   2. Element 1 — the specialization value (the "key"): an attribute that may trigger
  #     a separately compiled kernel variant. Currently possible values are:
  #     - None — no runtime specialization (used for bools, floats, do_not_specialize
  #         params)
  #     - "D" — value/pointer is divisible by 16 (alignment hint)
  #     - "" (empty string) — value/pointer is NOT divisible by 16
  #     - The actual Python value itself — for constexpr parameters and integer-valued 1
  #     - A cache_key — for JITCallable (nested kernel) arguments. Note that in that
  #         case the specialization value is a hash string (hopefully!!! lowercase).
  #         Likely by a coincidence this doesn't hurt to `attrs` building later!
  # The specialization values are determined at call time by native_specialize_impl in
  # C++ (triton/python/src/specialize.cc).
  # Specialization serves 3 goals: (1) affect kernel caching key, (2) discover additional
  # vars to be turned into constexprs by the compiler, (3) source for `attrs` spec.
  #
  # Their implementation handle params better, but we likely can't use their code since
  # they supply an argument value to specialize_impl() directly, while we can't do it
  # due to how jit compilation with traced values work, which doesn't even have actual
  # argument values, that's why the below code uses fake value of types.SimpleNamespace.
  # Also we only specialize runtime arguments (not constexprs!), while their impl does
  # it for everything.

  signature = {fn.arg_names[i]: v for i, v in enumerate(arg_dtypes)}
  # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
  # We assume that all arrays are aligned to 16 bytes, and Triton may use this
  # assumption, unless array args are include in the `do_not_specialize` list.
  #
  # Note that alignments, specialization and attrs (which are essentially alignment
  # optimization hints now) make sense and are computed only for positional/non-constexpr
  # arguments of the kernel! For constexprs compiler doesn't need this, as it has the
  # value itself.
  alignments = [16] * len(arg_dtypes)
  for i in scalar_args:
    alignments[i] = 0
  specialize_impl = _triton.native_specialize_impl
  is_const = False  # set as tl.const type hints and marks read-only data
  do_specialize = True
  specialization = tuple(
      specialize_impl(
          backend,
          types.SimpleNamespace(
              data_ptr=lambda: alignment, dtype=arg_dtype.removeprefix("*")
          ) if alignment>0 else arg_value,
          is_const,
          do_specialize,
          alignment > 0,
      )
      for arg_dtype, alignment, arg_value in zip(arg_dtypes, alignments, args)
  )
  # TODO now by passing actual value for a scalar it might be out of sync with how we
  # process constexprs. Fix this by improving specialization and getting constexpr status from it.

  attrs = {
      (i,): backend.parse_attr(attr)
      for i, (_, attr) in enumerate(specialization) if isinstance(attr, str)
      # attr could be None if specialization is disabled, or when an arg value is None.
      # Attributes for these doesn't make sense, and the upstream uses a similar check.
  }
  # attrs keys are tuples of integers representing a path into the (possibly nested)
  # kernel argument list as it is seen by the Triton compiler (i.e. including all
  # constexprs and whatnot, - everything as in the signature)

  constexpr_names = [p.name for p in fn.params if p.is_constexpr]

  constexprs = { k:v for k,v in metaparams.items() if k in constexpr_names}
  # note that original Triton passes _all_ arguments (including constexprs) through the
  # specialization pipeline and that could turn certain runtime vars with value of None
  # or 1 into constexprs. And then for the purposes of kernel compiling it decides what
  # constexpr and what isn't from the specialization results only!
  # We are short-cutting this, but perhaps we shouldn't, since it might make the
  # behavior incoherent. Also the short-cutting requires us to put constexprs into
  # kernel caching key due to that

  # None args are treated as constexprs in the original specialize.cc:L481
  # args==1 are also treated as constexprs IFF specialization is enabled in
  # specialize.cc:L238. The rationale is that 1s are extremely common as strides or
  # sizes, so it's very beneficial to compile that as a special case
  constexprs.update({fn.arg_names[i]: v for i, v in scalar_args.items() if v == 1 or v is None})

  # adding constexprs info to the signature
  for constant in constexprs:
    signature[constant] = "constexpr"

  # TODO here's what we need eventually to properly cache, compile and launch a kernel:
  # - kernel func signature dict {kernel param name -> data type} built from a rt made
  #     specialization
  # - `attrs` dict built from a runtime made specialization list
  # - constexprs dict {kernel param name -> constexpr value}, built from explicit
  #     metaparams, default values, and a runtime made specialization list
  # - backend options dict, built exclusively from metaparams passed
  # - renewed args + args_dtypes lists (having specialized constexprs removed and
  #     defaults added).
  # 
  # Intermediates:
  # - runtime made specialization list.

  # NOTE: To properly add defaults to rt-scalars requires:
  # - adding a default value to output arg too. We could have a convention to set it to
  #   None and verify that in the binder generating method during signature inspection.
  # - args and args_dtypes lists in extract_avals() can no longer be built from
  #   positional args (ctx.avals..) directly, but must be built from inspecting the
  #   signature and explicitly matching `strictly_out_avals` to the last
  #   `len(strictly_out_avals)` parameters backwards from the first constexpr argument.
  # - we must call generated binder function strictly using key-value arg notation to
  #   allow for correct argument matching.
  # - the generated binder function must check that none of outputs must have a value of
  #   None
  # Triton gets away with that by not requiring the outputs go last in the list of rt
  # arguments, so absent rt args get their defaults naturally (similarly no issues
  # to build named_args=dict(zip(self.arg_names, args)) in the autotuner - it just lacks
  # rt args with defaults to be set later). We (at this moment) must
  # specify outputs and they are always after in and in/out args, so to let the binder
  # match param names, we must call it using key-value representation of args...

  # TODO: generated binder function. Care for output arguments, JAX tracers and scalars.
  #  We need from it:
  
  #
  # TODO: extract constexprs from rt-args specialization too
  # 
  # TODO: return which args must be skipped from sending to the kernel due to turning
  # into constexprs.

  # TODO: 2 phase cache key resolution and store all that in the `fn` object itself.
  # Cache must include everything affecting
  # compilation, i.e.: specializations for runtime args, list of constexprs, backend
  # options, compute capability... what else?
  # Param datatypes/signature is already in specialization and constexprs

  # Cache key should contain any parameter that can affect the compiler output.
  cache_key = (
      fn,
      # tuple(signature.items()),
      specialization,

      # TODO fix this either by using a full specialization, or shorten to only use
      # added to constexprs vars
      tuple(constexprs.items()),
      #num_warps,
      #num_stages,
      #num_ctas,
      compute_capability,
      #enable_fp_fusion,
      #tuple(metaparams.items()),
      str(metaparams), # might be unhashable. Upstream does str()
  )
  kernel = _COMPILED_KERNEL_CACHE.get(cache_key)

  if kernel is None:
    # First, check that the kernel signature and the reconstructed signature have the
    # same number of parameters. A mismatch can occur due to differences in
    # `triton_call(input_output_aliases=)` handling between jax-triton versions.
    if len(fn.signature.parameters) != len(signature):
      raise TypeError(
        f"Number of parameters in the kernel '{fn}' signature "
        f"({len(fn.signature.parameters)}: {fn.signature}) "
        f"does not match reconstructed signature ({len(signature)}: {signature}). "
        "If the kernel was working on an older version of jax-triton and its "
        "triton_call() launcher uses `input_output_aliases` argument, note that "
        "implicit output arguments are no longer required for aliased args."
      )

    backend_fields = _BACKEND_OPTIONS_FIELD_NAMES[gpu_target.backend]
    opts = { k:v for k,v in metaparams.items() if k in backend_fields}

    options = backend.parse_options(opts)

    kernel_hash = abs(hash(cache_key))
    if _JAX_TRITON_DUMP_DIR:
      os.makedirs(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}")
      with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/config", "w") as f:
        pprint.pprint(cache_key, stream=f)
        pprint.pprint(options, stream=f)

    context = _triton.ir.context()
    _triton.ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)

    real_ASTSource = (
      gl_runtime.GluonASTSource
      if isinstance(fn, gl_runtime.GluonJITFunction)
      else tc.ASTSource
    )
    module = real_ASTSource(
      fn, constexprs=constexprs, signature=signature, attrs=attrs
    ).make_ir(gpu_target, options, codegen_fns, backend.get_module_map(), context)

    ttir = str(module)

    compilation_result = compile_ttir_inplace(
        module, backend, options, compute_capability, platform
    )

    kernel_name = compilation_result.name
    if _JAX_TRITON_DUMP_DIR:
      with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttir", "w"
      ) as f:
        f.write(ttir)
      with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ptx", "w"
      ) as f:
        f.write(compilation_result.binary)
      with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttgir", "w"
      ) as f:
        f.write(compilation_result.ttgir)
      with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.llir", "w"
      ) as f:
        f.write(compilation_result.llir)
      with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.compile_info",
          "w",
      ) as f:
        f.write(
            f"{kernel_name}: shared_mem_bytes:"
            f" {compilation_result.shared_mem_bytes}\n"
        )

    kernel = triton_kernel_call_lib.TritonKernel(
        kernel_name,
        num_warps,
        num_ctas,
        compilation_result.shared_mem_bytes,
        compilation_result.binary,
        ttir,
        compute_capability,
    )

    _COMPILED_KERNEL_CACHE[cache_key] = kernel

  return kernel, attrs


def make_autotuner_configs(
  fn: autotuner.Autotuner,
  scalar_args: dict[int, Any],
  metaparams: dict[str, Any],
  named_args: dict[str, Any],
) -> list[triton.Config]:
  """ Make and prune redundant autotuner configs based on user-provided metaparams.

  If any metaparams have been specified explicitly, we prune any configs that conflict.
  The pruning serves a specific need in jax-triton's architecture: unlike native Triton
  where autotuning happens dynamically at kernel launch, jax-triton must decide at
  lowering/tracing time which configs to compile. If the user has already fixed certain
  metaparameters (e.g., num_warps=4, BLOCK_SIZE=128), there's no point compiling or
  benchmarking autotuner configs that specify different values for those same
  parameters. The pruning eliminates those contradictory configs, reducing compilation
  and benchmarking work.

  Note that our implementation is more permissive than Triton's autotuner
  implementation, which will throw an error if any keys match.
  """
  assert isinstance(fn, autotuner.Autotuner)

  key_idxs = [fn.arg_names.index(k) for k in fn.keys]
  if any(idx not in key_idxs for idx in scalar_args):
    logging.warning(
        "Auto-tuning key does not include all scalar arguments. "
        "We may perform redundant auto-tuning."
    )

  prev_early_config_prune_fn = fn.early_config_prune

  def prune_configs(configs, named_args, **kwargs):
    pruned_configs = []
    for config in configs:
      if config.pre_hook is not None:
        raise NotImplementedError("`pre_hook` is not supported")

      # Keep the config IFF for every user-provided metaparam (k, v), the config
      # either doesn't specify k at all, or specifies the same value v. This ensures
      # the config is coherent with explicit user choices
      if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
        pruned_configs.append(config)
    if prev_early_config_prune_fn is not None:
      pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
    return pruned_configs

  fn.early_config_prune = prune_configs
  fn.nargs = named_args
  # TODO(Arech) if user has specified perf_model or early_config_prune functions in
  # prune_configs_by dict when was instantiating the autotuner, prune_configs() will
  # call each of it passing named_args to it. However, internally, the autotuner
  # constructs .nargs using only positional arguments provided in a similar way, so at
  # least currently there seems to be no divergence.
  configs = fn.prune_configs(metaparams)
  return configs


def make_configs_from_heuristics(
  fn: autotuner.Heuristics,
  configs: list[triton.Config],
  metaparams: dict[str, Any],
  named_args: dict[str, Any],
) -> list[triton.Config]:
  """Applies heuristics to the configs and returns the updated configs and the function."""
  assert isinstance(fn, autotuner.Heuristics)

  updated_configs = []
  for config in configs:
    kwargs = config.kwargs.copy()
    for name, heuristic in fn.values.items():
      kwargs[name] = heuristic({**named_args, **metaparams, **kwargs})
    updated_config = copy.copy(config)
    updated_config.kwargs = kwargs
    updated_configs.append(updated_config)
  return updated_configs


class JTJITFunction:
  """A wrapper around Triton's JITFunction/GluonJITFunction object to isolate the rest
  of the code from Triton's internals and provide a unified interface to bits needed.

  Additionally, it provides a persistence layer to ensure that a certain data doesn't
  have to be re-created on each kernel launch. A user may assume that even when they
  create a new JTJITFunction object for an object previously used JITFunction object,
  the persistent data is reused.

  Since we don't instantiate JTJITFunction objects the way JITFunction objects are
  instantiated and JTJITFunction have a short lifes, we have to store the data in the
  very JITFunction object itself. Once we implement a custom decorator similar to
  triton.jit() to have a JTJITFunction with a proper lifetime, this will be fixed.
  """

  def __init__(
    self,
    fn: autotuner.Heuristics
    | autotuner.Autotuner
    | triton.JITFunction
    | gl_runtime.GluonJITFunction,
  ):
    # peel off wrappers to get to the JITFunction object
    self.has_autotuner = isinstance(fn, autotuner.Autotuner)
    if self.has_autotuner:
      fn = fn.fn
    self.has_heuristics = isinstance(fn, autotuner.Heuristics)
    if self.has_heuristics:
      fn = fn.fn

    if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
      raise ValueError(
          "`kernel` must be a Triton's `JITFunction`, `GluonJITFunction`, `Heuristics` or `Autotuner` object."
      )
    self.fn = fn

  @cached_property
  def arg_names(self) -> list[str]:
    # .arg_names as per comment is deprecated
    return (
      self.fn.arg_names
      if hasattr(self.fn, "arg_names")
      else [p.name for p in self.fn.params]
    )


class KernelArgs:
  """Info on all non-constexpr arguments used to launch a kernel in one place.

  Note the semantic: function parameters are what are written in the function signature,
  while arguments are what is passed to function on its invocation.

  Each object has the following attributes:
  - scalars: dict[int, Any] - maps positional scalar args indices to values
  - input_output_aliases: dict[int, int] - maps input ordinal numbers to output ordinal numbers
  - args: list[Any] - list of all arguments (in, in-out, out), including scalars,
      in the order they are passed to the kernel.
  - arg_dtypes: list[str] - list of all arguments' Triton types in the order they are passed to the kernel
  - n_outputs: int - number of output arguments

  By construction, all arguments that aren't scalar_args have JAX abstract values.

  scalars, args and arg_dtypes might be updated during processing to add missing
  arguments with default values and to remove arguments that the compiler turns into
  constexprs.
  """

  def __init__(self, ctx, scalar_args: dict[int, Any], input_output_aliases: dict[int, int]):
    args, arg_dtypes = self._extract_avals(ctx, scalar_args, input_output_aliases)

    self.args = args
    self.arg_dtypes = arg_dtypes
    self.scalars = scalar_args
    self.input_output_aliases = input_output_aliases
    self.n_outputs = len(ctx.avals_out)

    # TODO: to properly provide defaults, we need a param_name->(value,dtype) mapping

  @staticmethod
  def _extract_avals(
    ctx,
    scalar_args: dict[int, Any],
    input_output_aliases: dict[int, int],
  ) -> tuple[list[Any], list[str]]:
    """ Extract all (input, scalar & output) arguments' abstract values and their Triton
    types from the context."""
    args = list(ctx.avals_in)
    arg_dtypes = list(map(get_triton_type, ctx.avals_in))
    for idx, v in scalar_args.items():
      args.insert(idx, v)
      arg_dtypes.insert(idx, "constexpr" if v is None else get_triton_type(v))

    # Extract only the output avals not referenced in the input_output_aliases mapping.
    strictly_out_avals = [
      aval
      for i, aval in enumerate(ctx.avals_out)
      if i not in input_output_aliases.values()
    ]
    args.extend(strictly_out_avals)
    arg_dtypes.extend(map(get_triton_type, strictly_out_avals))

    return args, arg_dtypes



def make_kernel_params(
  ctx,
  num_scalar_args: int,
  metaparams: dict[str, Any],
  grid,
  zeroed_outputs,
  input_output_aliases: dict[int, int],
  configs: list[triton.Config],
) -> list[dict[str, Any]]:
  """Make kernel call parameters for each config."""
  output2input = {v: k for k, v in input_output_aliases.items()}
  if len(output2input) != len(input_output_aliases):
    raise ValueError("input_output_aliases must be a bijection")

  outputs_offset = len(ctx.avals_in) + num_scalar_args
  zeroed_outputs_callable = callable(zeroed_outputs)

  kernel_params = []
  for config in configs:
    config_metaparams = {**metaparams, **config.kwargs}
    # filling back backend related config params
    config_metaparams["num_warps"] = config.num_warps
    config_metaparams["num_stages"] = config.num_stages
    config_metaparams["num_ctas"] = config.num_ctas
    if config.maxnreg is None:
      if "maxnreg" in config_metaparams:
        del config_metaparams["maxnreg"]
    else:
      config_metaparams["maxnreg"] = config.maxnreg

    config_grid = normalize_grid(grid, config_metaparams)

    config_zeroed_outputs = (
      zeroed_outputs(config_metaparams) if zeroed_outputs_callable else zeroed_outputs
    )

    # zeroed_params_with_sizes is a dict output_arg_idx -> aval_size_bytes
    # config_zeroed_outputs is a list of ordinal numbers of output arguments
    zeroed_params_with_sizes = {
      output2input[i] if i in output2input else i + outputs_offset: aval_size_bytes(
        ctx.avals_out[i]
      )
      for i in sorted(config_zeroed_outputs)
    }
    # TODO turning zeroed_params_with_sizes keys to _array_ indices only (i.e.
    # rely only on ctx.avals_in) will help get rid of dependency on scalars number
    # completely. But this requires "subtracting" relevant scalar numbers from
    # output2input values too (doable as we know all of that). Then we'd need to modify
    # `triton_kernel_call_lib.create_array_parameter()` call below.

    kernel_params.append(
      dict(
        metaparams=tuple(sorted(config_metaparams.items())),
        # num_warps=config.num_warps,
        # num_stages=config.num_stages,
        # num_ctas=config.num_ctas,
        grid=config_grid,
        zeroed_params_with_sizes=tuple(zeroed_params_with_sizes.items()),
      )
    )
  return kernel_params


def triton_kernel_call_lowering(
    make_gpu_target_func,
    ctx,
    *abstract_args,
    fn,
    concrete_args_tuple,
    kernel_call_name,
    custom_call_target_name,
    # out_shapes,
    grid,
    compute_capability,
    input_output_aliases,
    zeroed_outputs,
    serialized_metadata,
    args_kwargs,
    # metaparams: tuple[tuple[str, Any], ...],
):
  # we have to pass metaparams dictionary as a tuple to allow hashing necessary for
  # lowering via xla_primitive_callable()
  metaparams = from_hashable_metaparams(metaparams)

  assert isinstance(input_output_aliases, tuple), "input_output_aliases must be a tuple"
  input_output_aliases = dict[int,int](input_output_aliases)

  assert isinstance(concrete_args_tuple, tuple), "concrete_args_tuple must be a tuple"
  scalar_args = dict[int, Any](concrete_args_tuple)

  # if it fires, it's a programmer's mistake and no exception handling could fix that.
  # This is basically to validate the fields were set earlier in triton_call()
  assert all(
    isinstance(metaparams[k], int) for k in ("num_warps", "num_stages", "num_ctas")
  ), "num_warps, num_stages, and num_ctas must be integers"

  # TODO
  args, arg_dtypes = KernelArgs._extract_avals(ctx, scalar_args, input_output_aliases)
  # args here mean only positional i.e. non-constexpr args, including scalars.

  if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
    # named_args are needed only for Autotuner/Heuristics handling.
    #
    # Warning: fn.arg_names is deprecated and is scheduled for removal everywhere as per
    # comment in JITFunction code. It's however a part of public API of the Autotuner
    # and Heuristics classes, and it is unclear how they are going to get rid of it.
    # Doesn't exclude though that this still might break eventually.
    named_args = dict(unsafe_zip(fn.arg_names, args))
    # note that the above construction of named_args is based on positional args only.
    # This is fully coherent with how upstream does it in Autotuner/Heuristics classes.
    # These are genuinely name-value pairs for passed positional arguments only, and not
    # `bound_args` constructed using kernel's signature.

    # TODO: consequences of adding defaults to RT args here?

  if isinstance(fn, autotuner.Autotuner):
    # We should unpack autotuner first and heuristics second to ensure we'd
    # eventually get to a correct actual kernel `fn`.
    configs = make_autotuner_configs(fn, scalar_args, metaparams, named_args)
    fn = fn.fn
  else:
    ttcfg_args = dict(
      num_warps=metaparams["num_warps"],
      num_stages=metaparams["num_stages"],
      num_ctas=metaparams["num_ctas"],
    )
    if "maxnreg" in metaparams:
      ttcfg_args["maxnreg"] = metaparams["maxnreg"]
    configs = [triton.Config({}, **ttcfg_args)]

  if isinstance(fn, autotuner.Heuristics):
    configs = make_configs_from_heuristics(fn, configs, metaparams, named_args)
    fn = fn.fn

  if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
    raise ValueError(
        "`kernel` must be a Triton `JITFunction`, `GluonJITFunction`, `Heuristics` or `Autotuner`."
    )

  kernel_params = make_kernel_params(
    ctx,
    len(scalar_args),
    metaparams,
    grid,
    zeroed_outputs,
    input_output_aliases,
    configs,
  )

  kernel_calls = []
  for params in kernel_params:
    kernel, specialization_attr = get_or_create_triton_kernel(
      make_gpu_target_func,
      ctx.module_context.platforms[0],
      fn,
      args,
      arg_dtypes,
      scalar_args,
      # num_warps=params["num_warps"],
      # num_stages=params["num_stages"],
      # num_ctas=params["num_ctas"],
      compute_capability=compute_capability,
      # enable_fp_fusion=metaparams["enable_fp_fusion"],
      metaparams=dict(params["metaparams"]),
      # debug=metaparams["debug"],
    )

    call_params = []
    zeroed_params_with_sizes = dict(params["zeroed_params_with_sizes"])

    # TODO how does it handle positional scalars in random positions?

    equal_to_1_or_None = {i for i, v in scalar_args.items() if v == 1 or v is None}
    for i, (arg, dtype) in enumerate(zip(args, arg_dtypes)):
      if isinstance(arg, core.ShapedArray):
        arg_attrs = specialization_attr[(i,)]
        call_params.append(
            triton_kernel_call_lib.create_array_parameter(
                zeroed_params_with_sizes.get(i, 0),
                16 if (["tt.divisibility", 16] in arg_attrs) else 0,
            )
        )
      elif i not in equal_to_1_or_None:
        # Convert TypedInt/TypedFloat subclasses to plain Python types,
        # as nanobind's strict-mode integer caster rejects subclasses.
        if isinstance(arg, bool):
          arg = bool(arg)
        elif isinstance(arg, int):
          arg = int(arg)
        elif isinstance(arg, (float, np.float32, np.float64)):  # np dtypes are added
          # to handle possible default values properly too
          arg = float(arg)
        call_params.append(
            triton_kernel_call_lib.create_scalar_parameter(arg, dtype)
        )

    kernel_calls.append(
        triton_kernel_call_lib.TritonKernelCall(
            kernel,
            params["grid"][0],
            params["grid"][1],
            params["grid"][2],
            call_params,
        )
    )

  if len(kernel_calls) > 1:
    # named_scalar_args seems to be only used for naming of the autotune call, which is
    # used only for logging, so it's not important how it handles scalar_args with 1s
    # and None-s, but still might just need a better name
    named_scalar_args = {fn.arg_names[i]: v for i, v in scalar_args.items()}
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in input_output_aliases.items()
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
        f"{kernel_call_name} ({fn.fn.__name__}) {named_scalar_args}",
        [(call, str(config)) for call, config in zip(kernel_calls, configs)],
        input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)
  rule = jax.ffi.ffi_lowering(
      custom_call_target_name,
      api_version=2,
      backend_config=zlib.compress(call_proto),
      operand_output_aliases=input_output_aliases,
  )
  # Note if needed,array_args could be permuted before passing to the rule() 
  return rule(ctx, *abstract_args)


mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_gpu_target_cuda),
    platform="cuda",
)

mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_gpu_target_hip),
    platform="rocm",
)


def triton_kernel_call_raise_on_jvp(*args, **kwargs):
  del args, kwargs  # unused
  raise NotImplementedError(
      "jax_triton.triton_call does not support automatic differentiation. Use "
      "jax.custom_jvp or jax.custom_vjp to implement a custom automatic "
      "differentiation rule for your kernel."
  )


ad.primitive_jvps[triton_kernel_call_p] = triton_kernel_call_raise_on_jvp


def triton_kernel_call_raise_on_vmap(*args, **kwargs):
  del args, kwargs  # unused
  raise NotImplementedError(
      "jax_triton.triton_call does not support batching with jax.vmap. Use "
      "jax.custom_batching.custom_vmap to implement a custom batching rule for "
      "your kernel."
  )


batching.primitive_batchers[triton_kernel_call_p] = (
    triton_kernel_call_raise_on_vmap
)


class ShapeDtype(Protocol):

  @property
  def shape(self) -> tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...


def to_hashable_metaparams(metaparams: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
  """Converts metaparams to a tuple so it could be hashable while caring for certain
  keys this are known to be unhashable."""
  # Triton doesn't support passing unhashable values as metaparams, however, these
  # metaparams might also contain backend options, and some of them are dicts. So far,
  # there's only one such option - `extern_libs`, both in AMD and NVIDIA backends.
  # While it might be possible to use `tree_util.tree_flatten(metaparams)` here, it's
  # too expensive for handling such a rare case. Instead, we do this manually.
  # This is somewhat fragile and might require some maintenance to adapt to new options
  # but the kernel launcher is a hot path, so this seem tolerable.
  if "extern_libs" in metaparams:
    metaparams["extern_libs"] = tuple(metaparams["extern_libs"].items())
  return tuple(metaparams.items())


def from_hashable_metaparams(astuple: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
  """Converts a tuple of metaparams back to a dictionary."""
  assert isinstance(astuple, tuple), "astuple must be a tuple of metaparam items"
  metaparams = dict(astuple)
  if "extern_libs" in metaparams:
    metaparams["extern_libs"] = dict(metaparams["extern_libs"])
  return metaparams


def serialize_args_kwargs(fn, args: list, kwargs: dict):
  """Prepares args and kwargs for passing through JAX's primitive system by separating
  things that must be traced from things that must be passed as is.

  Returns two sequences of traced values and a tuple of hashable reconstruction info.
  Assumes that a caller adheres to the Triton's rules about argument types, so no
  special checks for hashability are done for performance reasons.

  Note that the function consumes kwargs, i.e. its state is modified in place.
  """
  # JAX's Primitive.bind(*args, **params) has a strict dichotomy:
  # - *args (positional) are dynamic operands — values that participate in JAX's
  # tracing/transformation system (jit, grad, vmap). During JIT compilation, they become
  # abstract values (ShapedArray) and then MLIR SSA values in the lowered IR.
  # - **params (keyword) are static parameters — they must be hashable, are passed
  # through verbatim, and retain their concrete Python values across the entire
  # compilation pipeline.
  # Hence to pass data through JAX's primitive system we must separate things that must
  # be traced from things that must be passed as is. The restriction is that since JAX
  # arrays must be managed by XLA, they must be passed as traced items. The rest could
  # go concrete helping to specialize the kernel properly.
  # We must implement the separation in a way that (1) allows to reconstruct both `args`
  # and `kwargs` back exactly on the other end to process them properly, and (2) all JAX
  # array arguments go into .bind() call in exactly the same order as the kernel expects
  # them (so we don't have to reshuffle them later during the lowering). In that we
  # assume that no JAX array could be specialized to a constexpr for compiling, which
  # holds in the upstream Triton too (small arrays could be passed as tuples/lists
  # though to constexpr annotated params).

  flat_args, args_tree = tree_util.tree_flatten(args, is_leaf=lambda x: x is None)
  # We must let Nones to pass through flattening and `is_leaf` semantic is additive.

  # Since there's usually much more non-vector parameters than vectors, extracting
  # vectors from flat_args. Jax explicitly guarantee that `isinstance(x, jnp.ndarray)`
  # check behaves correctly for raw arrays as well as for tracers. For jitting all/most
  # of Python scalars are mapped to ShapedArray avals too, so they are treated as arrays
  # by the check. However, there's no use-case where we might want to pass a scalar as a
  # traceable thing into the kernel call, so we can safely assume a user will mark it as
  # a static argument for jitting (the old implementation relied on that too).
  abs_args = {i: v for i, v in enumerate(flat_args) if isinstance(v, jax.Array)}
  static_args = (v for v in flat_args if not isinstance(v, jax.Array))
  # we're going to pass flat_args as static params now. There's a caveat at least for
  # tl.constexpr() objects: JAX's lowering code seems to require only hashability
  # and correct equality semantics for static objects, and this is true for
  # tl.constexpr() with a nuance: on comparison it returns not a bool True/False, but
  # another tl.constexpr() object with a bool value. So if there's a silly check in JAX
  # that the `(a==b) is True`, or `type(a==b) is bool` - it'll break.

  # Now do the same for kwargs with a caveat - we must traverse them in order of kernel
  # parameters declaration to ensure arrays ends up in a correct order
  def contain_arrays(v):
    if isinstance(v, jax.Array):
      return True
    elif isinstance(v, tuple):  # among pytrees only tuples could be passed as arguments
      return any(contain_arrays(x) for x in v)
    return False

  class _FakeVal: ...

  abs_kwargs_list = []  # only values to abstract
  abs_kwargs_meta = {}  # hashable reconstruction info, keyed by a parameter name
  for aname in JTJITFunction(fn).arg_names:
    # if kwargs has that parameter AND the value contains an array somewhere (including
    # nested tuples/lists), we handle the whole arg value differently to be able to
    # reconstruct it correctly. Otherwise we just skip the arg to pass it along with
    # other kwargs, such as backend options.
    arg_val = kwargs.get(aname, _FakeVal())
    if contain_arrays(arg_val):
      del kwargs[aname]  # handle the value differently
      if isinstance(arg_val, jax.Array):  # shortcutting if the value is a raw array
        abs_kwargs_list.append(arg_val)
        abs_kwargs_meta[aname] = None
      else:  # do full flattening
        flat_v, the_tree = tree_util.tree_flatten(arg_val)
        abs_v = {i: v for i, v in enumerate(flat_v) if isinstance(v, jax.Array)}
        static_v = tuple(v for v in flat_v if not isinstance(v, jax.Array))
        abs_kwargs_list.extend(abs_v.values())
        abs_kwargs_meta[aname] = (static_v, tuple(abs_v.keys()), the_tree)

  return (
    abs_args.values(),  # intentionally not materialized
    abs_kwargs_list,
    (  # first args info
      args_tree,
      tuple(abs_args.keys()),
      tuple(static_args),
      # kwargs info
      tuple(abs_kwargs_meta.items()),
      to_hashable_metaparams(kwargs),
    ),
  )


def deserialize_args_kwargs(
  ctx, abstract_args: list, args_kwargs_meta: tuple
) -> tuple[list, dict]:
  """Reconstructs args and kwargs from the serialized form. Abstract arguments are
  replaced by stubs."""
  args_tree, abs_args_keys, static_args, abs_kwargs_meta, kwargs = args_kwargs_meta
  abs_kwargs_meta = dict[str, None | tuple[tuple, tuple, Any]](abs_kwargs_meta)
  # abstract_args is a concatenation of abs_args.values() and abs_kwargs_list, but 
  # abs_kwargs_meta has a more complex inner structure and its length could be smaller
  # than a corresponding portion of abstract_args storing abs_kwargs_list.

  default_alignment = 16
  def make_stub(i: int):
    arg_dtype = get_triton_type(ctx.avals_in[i])
    return types.SimpleNamespace(
      data_ptr=lambda: default_alignment, dtype=arg_dtype.removeprefix("*")
    )

  args = list(static_args)
  kwargs_ofs = len(abs_args_keys)
  abs_v = [make_stub(i) for i in range(kwargs_ofs)]
  for i,v in unsafe_zip(abs_args_keys, abs_v):
    args.insert(i, v)
  args = tree_util.tree_unflatten(args_tree, args)

  kwargs = from_hashable_metaparams(kwargs)
  for aname, meta in abs_kwargs_meta.items():
    assert aname not in kwargs
    if meta is None:
      kwargs[aname] = make_stub(kwargs_ofs)
      kwargs_ofs += 1
    else:
      static_v, abs_v_keys, the_tree = meta
      n_keys = len(abs_v_keys)
      abs_v = [make_stub(kwargs_ofs + j) for j in range(n_keys)]
      kwargs_ofs += n_keys
      static_v = list(static_v)
      for k,v in unsafe_zip(abs_v_keys, abs_v):  # assume sorted keys
        static_v.insert(k, v)
      kwargs[aname] = tree_util.tree_unflatten(the_tree, static_v)
  assert kwargs_ofs == len(abstract_args)

  return args, kwargs


def triton_call(
  *args: jax.Array | bool | int | float | np.float32,
  kernel: (
    triton.JITFunction
    | gl_runtime.GluonJITFunction
    | triton.runtime.Heuristics
    | triton.runtime.Autotuner
  ),
  out_shape: ShapeDtype | Sequence[ShapeDtype],
  grid: GridOrLambda,
  name: str = "",
  custom_call_target_name: str = "triton_kernel_call",
  num_warps: int | None = None,
  num_stages: int | None = None,
  num_ctas: int = 1,  # TODO(giorgioa): Add support for dimensions tuple.
  compute_capability: int | None = None,
  enable_fp_fusion: bool = True,
  input_output_aliases: dict[int, int] | None = None,
  zeroed_outputs: (Sequence[int] | Callable[[dict[str, Any]], Sequence[int]]) = (),
  debug: bool = False,
  serialized_metadata: bytes = b"",
  **kwargs: Any,
) -> Any:
  """Calls a Triton kernel with `jax.Array` arguments.

  Example usage:

  First we define a simple kernel that adds two vectors.

  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def add_kernel(
      x_ptr,
      y_ptr,
      output_ptr,
      block_size: tl.constexpr,
  ):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < 8
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  ```

  Then we use `triton_call` to call it from JAX.

  ```python
  import jax
  import jax.numpy as jnp
  import jax_triton as jt

  def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 8
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_shape=out_shape,
        grid=(x.size // block_size,),
        block_size=block_size)

  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))
  ```

  Args:
    *args: Positional inputs for the Triton kernel. All input arrays must be passed as
      positional arguments in the order they are declared in the kernel signature,
      except for purely output arguments that are allocated and passed to the kernel
      implicitly (user do not need to pass them explicitly). In the kernel signature,
      purely output parameters should be anywhere after the last input array parameter.
      Scalar values can be passed as positional arguments in this sequence, or as
      key-value arguments.
    kernel: A Triton kernel (e.g. a function decorated with `triton.jit`). All
      static values should be annotated with `triton.language.constexpr` or
      `triton.experimental.gluon.language.constexpr`.
    out_shape: A `jax.ShapeDtypeStruct` (or something that has `.shape` and
      `.dtype` attributes) or a sequence thereof that specify the output(s) of
      the kernel. Pointers for each of the `jax.ShapeDtypeStruct`s in
      `out_shape` will be passed into `kernel` following the input parameters.
    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invocated in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `**kwargs` and should return a
      tuple of up to 3 integers.
    input_output_aliases: A dictionary mapping input argument indices in `args`
      to output argument indices, to alias the corresponding buffers.
    zeroed_outputs: A sequence of output indices, or a function returning a sequence of
      such indices, for outputs that should be zeroed before the kernel is launched.
      This argument also supports zeroing input-output (i.e. aliased through
      `input_output_aliases`) arguments.
    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    num_ctas: The size of thread blocks per cluster to be used on GPUs with
      compute capabilities >= 9.0. It must be less or equal to 8.
    enable_fp_fusion: Whether to enable floating-point operands fusion for the kernel.
    debug: Prints out intermediate IRs if True for debugging purposes. Also used as a
      the backend options argument.
    serialized_metadata: Arbitrary metadata that will be added into the
      serialized kernel call.
    kwargs: Key-value pairs (num_warps, num_stages, num_ctas, debug and enable_fp_fusion
      arguments are added there automatically) that will be provided to:
      - a `grid` (if it is a function),
      - backend options constructor (only recognized arguments are passed),
      - the Triton kernel as `constexpr` arguments (constexprs must always be scalars)
        or regular runtime arguments (scalars only; arrays must be passed as positional
        arguments) (only recognized as kernel parameters are passed).

    Default values in kernel signature are supported only for scalars, or for output
    arguments (for outputs only the value of None is allowed as a default - this is
    useful when a scalar with a default value needs to precede an output argument).

  Returns:
    Outputs from the Triton kernel.
  """
  if input_output_aliases is None:
    input_output_aliases = {}

  # TODO support for outputs naming
  out_shape = tree_util.tree_map(
    lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape
  )
  # TODO(sharadmv): check in_tree is flat (no Pytrees allowed in triton_call)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

  # Python guarantees the keys don't exist. The original Triton has a single namespace
  # for both constexprs and backend options, and we're doing the same to unify processing
  # We are setting defaults here early, since we use values early
  kwargs["num_warps"] = num_warps if num_warps is not None else 4
  kwargs["num_stages"] = num_stages if num_stages is not None else 3
  kwargs["num_ctas"] = num_ctas
  kwargs["enable_fp_fusion"] = enable_fp_fusion
  kwargs["debug"] = debug

  abs_args, abs_kwargs, args_kwargs_meta = serialize_args_kwargs(kernel, args, kwargs)

  out_flat = triton_kernel_call_p.bind(
    *abs_args,
    *abs_kwargs,
    fn=kernel,
    kernel_call_name=name,
    custom_call_target_name=custom_call_target_name,
    # out_shapes=tuple(flat_out_shapes),
    grid=grid,
    compute_capability=compute_capability,
    input_output_aliases=tuple(input_output_aliases.items()),
    zeroed_outputs=zeroed_outputs,
    serialized_metadata=serialized_metadata,
    # metaparams=to_hashable_metaparams(kwargs),
    args_kwargs=args_kwargs_meta,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
