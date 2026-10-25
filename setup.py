"""Compiled-extension build for the standalone `xtim` distribution.

Builds the pybind11 extension `xtim._xtim` from the bundled C++ engine sources
under cpp/, and the TEST-API extension `xtim._xtim_testapi` (cpp/testapi: the
exact record-law oracle) linked against the engine objects compiled for _xtim —
so _xtim itself is built exactly as without it. Metadata lives in pyproject.toml.

Build profile (mirrors the development engine):
  * GCC / Clang: -O3 -funroll-loops, plus -march=native UNLESS QECCORE_PORTABLE
    is set. Portable wheels (cibuildwheel) set QECCORE_PORTABLE=1 so the binary
    runs on any CPU and is bit-reproducible across machines. Conda's
    -fno-strict-overflow is stripped (it can flip last-bit float results).
  * MSVC: /O2 plus a force-included builtin shim (cl.exe has no __builtin_ctz/
    popcount/...); no -march analogue, i.e. a generic-x64 (portable) build.
"""
import os
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


_ENGINE_SOURCES = sorted(glob("cpp/src/*.cpp"))


# J-b2 J3.5 (fixplan-J4-unit-growth, ruled (A)): GCC's inline-unit-growth limit is a percentage of the unit size,
# so deleting dead code from the two twirl TUs lowered their inlining budget and moved hot codegen. A per-TU budget
# that never binds (verified with -fopt-info-inline-missed: 0 "inline-unit-growth limit reached") keeps their codegen
# a function of the live code only. GCC / Clang (non-MSVC) builds only.
_UNIT_GROWTH_TUS = ("cpp/src/twirl_sampler.cpp", "cpp/src/twirl_sampler_fam.cpp", "cpp/src/twirl_sampler_er.cpp")
_UNIT_GROWTH_ARGS = ["--param", "inline-unit-growth=400"]


class build_ext_xtim(build_ext):  # noqa: N801 - distutils naming convention
    _STRIP_FLAGS = {"-fno-strict-overflow"}

    def build_extensions(self):
        if self.compiler.compiler_type != "msvc":
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
        if ext.name == "xtim._xtim_testapi":
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
                args = base + ["-O3", "-funroll-loops"]
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


ext_modules = [
    Pybind11Extension(
        "xtim._xtim",
        _ENGINE_SOURCES + ["cpp/bindings/xtim_py.cpp"],
        include_dirs=["cpp/include"],
        cxx_std=17,
        # xtim_py.cpp #includes apps/ref_compile.cpp, which #includes
        # apps/protocol_reference.cpp (the shared gate battery). Those TU-#included .cpp
        # are NOT compilation units in the source list, so without listing them as
        # `depends` build_ext's timestamp check never rebuilds _xtim.so when they change
        # — the stale-.so footgun. List them so an edit to either forces a rebuild.
        depends=["cpp/apps/ref_compile.cpp", "cpp/apps/protocol_reference.cpp",
                 # header-only core of the compiled sector table (3.1.3), included by the
                 # bindings TU only — same footgun, same fix.
                 "cpp/include/qeccore/frame_cosets.hpp"],
    ),
    # S1 test API (not a public interface): the exact record-law oracle, own extension.
    Pybind11Extension(
        "xtim._xtim_testapi",
        sorted(glob("cpp/testapi/*.cpp")),
        include_dirs=["cpp/include"],
        cxx_std=17,
        depends=["cpp/testapi/record_law_oracle.hpp"] + _ENGINE_SOURCES,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext_xtim})
