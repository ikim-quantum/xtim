"""Compiled-extension build for the standalone `xtim` distribution.

Builds the pybind11 extension `xtim._xtim` from the bundled C++ engine sources
under cpp/, and the TEST-API extension `xtim._xtim_testapi` (cpp/testapi: the
exact record-law oracle) linked against the engine objects compiled for _xtim —
so _xtim itself is built exactly as without it. The test API is built ONLY by
in-place builds (`python setup.py build_ext --inplace`, `pip install -e .`); a
wheel (`pip wheel .`, cibuildwheel) never contains it, and the sdist carries no
cpp/testapi or cpp/apps. Metadata lives in pyproject.toml.

Build profile (mirrors the development engine):
  * GCC / Clang: -O3 -funroll-loops, plus -march=native UNLESS QECCORE_PORTABLE
    is set. Portable wheels (cibuildwheel) set QECCORE_PORTABLE=1 so the binary
    runs on any CPU and is bit-reproducible across machines. Conda's
    -fno-strict-overflow is stripped (it can flip last-bit float results).
    -ffp-contract=off always (see _FP_ARGS): exported DEM/xdem bytes do not depend on -march.
  * MSVC: /O2 plus a force-included builtin shim (cl.exe has no __builtin_ctz/
    popcount/...); no -march analogue, i.e. a generic-x64 (portable) build.
"""
import os
import subprocess
import sys
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


_ENGINE_SOURCES = sorted(glob("cpp/src/*.cpp"))


# GCC's inline-unit-growth limit is a percentage of the unit size,
# so deleting dead code from the two twirl TUs lowered their inlining budget and moved hot codegen. A per-TU budget
# that never binds (verified with -fopt-info-inline-missed: 0 "inline-unit-growth limit reached") keeps their codegen
# a function of the live code only. GCC builds only: clang has no such --param (it warns
# -Wunused-command-line-argument), and MSVC none either.
_UNIT_GROWTH_TUS = ("cpp/src/twirl_sampler.cpp", "cpp/src/twirl_sampler_fam.cpp", "cpp/src/twirl_sampler_er.cpp")
_UNIT_GROWTH_ARGS = ["--param", "inline-unit-growth=400"]


def _run(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


def _is_clang(cmd):
    """True when the compiler command `cmd` (argv list) is clang (Apple clang included).
    1. Preprocess an empty C++ input and look for the predefined __clang__, so a `gcc`/`cc`/`c++` name that is really
       clang (macOS) is caught and a real GCC is never misread. `-arch X` pairs are dropped from the probe: clang -E
       refuses several of them (a python.org universal2 Python without ARCHFLAGS passes -arch arm64 -arch x86_64).
    2. If that probe fails: `<compiler> --version` mentioning clang (GCC prints no such word).
    3. If both fail: clang on macOS (its only supported compiler), not clang elsewhere (GCC flags kept, as before)."""
    argv, skip = [], False
    for a in cmd:
        if skip:
            skip = False
        elif a == "-arch":
            skip = True
        else:
            argv.append(a)
    p = _run(argv + ["-dM", "-E", "-x", "c++", os.devnull])
    if p is not None and p.returncode == 0:
        return "__clang__" in p.stdout
    v = _run([cmd[0], "--version"])
    if v is not None and v.returncode == 0:
        return "clang" in v.stdout.lower()
    return sys.platform == "darwin"


# Floating-point contraction OFF in every GCC / Clang TU (CMake: the same flag). GCC contracts a*b+c into one FMA by
# default (C++ is never ISO-strict here), and only a -march=native build on an FMA CPU can emit FMA, so a native and a
# portable build printed different last digits of exported probabilities (e.g. xdem::xor_compose, P + p - 2 P p: one
# vfnmadd231sd natively). The export's call graph reaches ~all engine TUs, so the flag is global, not per TU. With it,
# every build evaluates the same IEEE basic operations in source order: the portable (no-FMA) build's code is
# unchanged and the native build agrees with it bit for bit. MSVC /fp:precise does not contract.
_FP_ARGS = ["-ffp-contract=off"]


class build_ext_xtim(build_ext):  # noqa: N801 - distutils naming convention
    _STRIP_FLAGS = {"-fno-strict-overflow"}
    _TESTAPI = "xtim._xtim_testapi"

    def finalize_options(self):
        super().finalize_options()
        # The test API is a development artifact: in-place builds only (setuptools sets inplace for
        # `pip install -e`). Wheels never ship it.
        if not self.inplace:
            self.extensions = [ext for ext in self.extensions if ext.name != self._TESTAPI]

    def build_extensions(self):
        cxx = getattr(self.compiler, "compiler_so_cxx", None) or getattr(self.compiler, "compiler_so", None)
        if self.compiler.compiler_type != "msvc" and not (cxx and _is_clang(cxx)):
            orig_compile = self.compiler._compile

            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if src.replace(os.sep, "/").endswith(_UNIT_GROWTH_TUS):
                    extra_postargs = list(extra_postargs) + _UNIT_GROWTH_ARGS
                return orig_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
            self.compiler._compile = _compile
        self._build_extensions_in_order()

    def _build_extensions_in_order(self):
        # In list order, never in parallel: _xtim_testapi links the engine objects _xtim compiles.
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if ext.name == self._TESTAPI:
            # The engine objects _xtim compiled (same flags), as a static archive: the linker
            # takes only what the oracle needs. _xtim's own sources, objects and link are untouched.
            objs = self.compiler.object_filenames(_ENGINE_SOURCES, output_dir=self.build_temp)
            self.compiler.create_static_lib(objs, "xtim_engine", output_dir=self.build_temp)
            ext.extra_objects = [self.compiler.library_filename("xtim_engine", output_dir=self.build_temp)]
        orig_so = getattr(self.compiler, "compiler_so", None)
        # APPEND to the existing args: Pybind11Extension(cxx_std=17) already injected
        # the platform-correct C++17 std flag (-std=c++17 / /std:c++17) here, and it
        # must be preserved (replacing the list drops it -> std::optional/variant fail).
        base = list(ext.extra_compile_args or [])
        try:
            if self.compiler.compiler_type == "msvc":
                # MSVC ignores the GCC/Clang flags; add /O2 and force-include the builtin
                # shim (resolved via -I cpp/include). No -march analogue -> generic x64.
                # /D_USE_MATH_DEFINES: MSVC otherwise leaves M_PI etc. undefined.
                ext.extra_compile_args = base + [
                    "/O2", "/FIqeccore/compiler_builtins.hpp", "/D_USE_MATH_DEFINES"]
            else:
                args = base + ["-O3", "-funroll-loops"] + _FP_ARGS
                if not os.environ.get("QECCORE_PORTABLE"):
                    args.append("-march=native")
                ext.extra_compile_args = args
                if orig_so is not None:
                    self.compiler.compiler_so = [
                        f for f in orig_so if f not in self._STRIP_FLAGS]
            super().build_extension(ext)
        finally:
            ext.extra_compile_args = base
            if orig_so is not None:
                self.compiler.compiler_so = orig_so


# The headers the sources include, as `depends`: distutils rebuilds an extension when one of them is newer than its
# .so, so a header-only edit (e.g. cpp/bindings/record_buffer.hpp) is never served by a stale extension.
_HEADERS = sorted(glob("cpp/include/qeccore/*.hpp") + glob("cpp/bindings/*.hpp") + glob("cpp/src/*.hpp"))

ext_modules = [
    Pybind11Extension(
        "xtim._xtim",
        _ENGINE_SOURCES + ["cpp/bindings/xtim_py.cpp"],
        include_dirs=["cpp/include"],
        cxx_std=17,
        depends=_HEADERS,
    ),
    # S1 test API (not a public interface): the exact record-law oracle and the exact engine's
    # entry points (exact-out T5), own extension. In-place builds only (build_ext_xtim.finalize_options).
    Pybind11Extension(
        "xtim._xtim_testapi",
        sorted(glob("cpp/testapi/*.cpp")),
        include_dirs=["cpp/include"],
        cxx_std=17,
        # exact_py.cpp #includes apps/ref_compile.cpp, which #includes apps/protocol_reference.cpp (the shared
        # gate battery): TU-#included .cpp, listed so an edit to either forces a rebuild (exact-out T5).
        # (+ the header-only core of the compiled sector table, frame_cosets_py.cpp: exact-out T6)
        # (+ the sampler's internal header: cpp/testapi/wire_rows_access.cpp #includes it, S2-FINALFIX F2)
        # (+ the GPU kernel source: cpp/testapi/xdem_gpu_ref_py.cpp #includes it, GPU Task 4)
        depends=["cpp/testapi/record_law_oracle.hpp", "cpp/testapi/xdem_lean_dense_oracle.hpp",
                 "cpp/testapi/xdem_sweep_dense_oracle.hpp", "cpp/apps/ref_compile.cpp",
                 "cpp/src/twirl_sampler_impl.inc.hpp",
                 "cpp/apps/protocol_reference.cpp", "cpp/include/qeccore/frame_cosets.hpp",
                 "xtim/gpu/xdem_kernels.h"] + _ENGINE_SOURCES
                + sorted(glob("cpp/testapi/*.hpp")) + _HEADERS,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext_xtim})
