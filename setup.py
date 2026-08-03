"""Compiled-extension build for the standalone `xtim` distribution.

Builds the single pybind11 extension `xtim._xtim` from the bundled C++ engine
sources under cpp/. Metadata lives in pyproject.toml.

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


class build_ext_xtim(build_ext):  # noqa: N801 - distutils naming convention
    _STRIP_FLAGS = {"-fno-strict-overflow"}

    def build_extension(self, ext):
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
        sorted(glob("cpp/src/*.cpp")) + ["cpp/bindings/xtim_py.cpp"],
        include_dirs=["cpp/include"],
        cxx_std=17,
        # xtim_py.cpp #includes apps/ref_compile.cpp, which #includes
        # apps/protocol_reference.cpp (the shared gate battery). Those TU-#included .cpp
        # are NOT compilation units in the source list, so without listing them as
        # `depends` build_ext's timestamp check never rebuilds _xtim.so when they change
        # — the stale-.so footgun. List them so an edit to either forces a rebuild.
        depends=["cpp/apps/ref_compile.cpp", "cpp/apps/protocol_reference.cpp"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext_xtim})
