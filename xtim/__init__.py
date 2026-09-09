"""xtim — Stim-shaped Python UX for the MSP simulator engine.

    import xtim
    c = xtim.Circuit.from_file("protocol.stim")
    dets, obs, exps = c.compile_detector_sampler(seed=7).sample(
        10**6, separate_observables=True, return_expectations=True)
    meas = c.compile_sampler(seed=7).sample(10**6)
    dem  = c.detector_error_model()      # a real stim.DetectorErrorModel
    info = c.reference_info()

Spec: docs/superpowers/specs/2026-06-12-xtim-python-ux-design.md.
The engine emits RAW physics only; decoding policy is user-side numpy.

`cache_dir` configures where the invisible reference cache lives (None = the
default `.xtim_cache/` under the current working directory):

    xtim.cache_dir = "/path/to/cache"
"""
from __future__ import annotations

# Single source of truth for the version is pyproject.toml; read it back from the
# installed package metadata. The dev fallback covers running from a source tree
# (the monorepo) where xtim is not pip-installed.
try:
    from importlib.metadata import version as _v

    __version__ = _v("xtim")
except Exception:  # not installed (source checkout) -> dev marker
    __version__ = "3.1.2+dev"
finally:
    globals().pop("_v", None)  # don't leak the import into the public namespace

# Where the invisible reference cache lives; None -> .xtim_cache/ under the cwd.
cache_dir: "str | None" = None

from ._xtim import ENGINE_VERSION, REF_FORMAT_VERSION  # noqa: E402
from .circuit import (  # noqa: E402
    Circuit,
    CompiledDetectorSampler,
    CompiledSampler,
    DemWithReject,
    PostselectFault,
    TwirlDetectorSampler,
)
from .collect import Task, collect, scale_noise  # noqa: E402
from .diagnose import Diagnosis, ExpectationInfo, diagnose  # noqa: E402
from .ler import PostselectedLER, postselected_logical_error_rate  # noqa: E402
from .examples import example_path, list_examples, load_example  # noqa: E402
from .errors import (  # noqa: E402
    XtimCacheWarning,
    XtimDemError,
    XtimError,
    XtimParseError,
    XtimPerformanceWarning,
    XtimRejectError,
    XtimReferenceError,
    XtimStimMissingError,
)
from .reference import Reference  # noqa: E402
from .fidelity_helpers import extract_frame, fidelity_from_logicals  # noqa: E402
# Port v3: declared-consumption segment surface (facade over existing samplers).
# Import as xtim.port and expose the two nouns + two verbs at the top level.
from . import port  # noqa: E402
from .port import Consume, compile as compile_segment, Segment, Result  # noqa: E402
# Engine-level twirl surface (the record/barrier/decision sampler generation +
# port-contract state objects). This is the public API consumed by adaptq's
# M2b/M3 validation suite; the orchestration layer (feedback/stage_api/tier/...)
# lives in the monorepo and is deliberately NOT part of the standalone package.
from .twirl import (  # noqa: E402
    compile_twirl_sampler,
    compile_twirl_sampler_from_state,
    bare_state_of,
    input_state_of,
    # Private aliases, re-exported deliberately and NOT in __all__: they are
    # the pre-2.x spellings, kept so existing notebooks importing the
    # underscore names keep working.  Unused inside this repo by design --
    # suppressed rather than deleted, since removing a name from a PUBLISHED
    # package is a breaking change and 3.0.0 has already shipped.
    _bare_state_of,  # noqa: F401
    _input_state_of,  # noqa: F401
    _project_bare_onto_syndrome,  # noqa: F401
)

__all__ = [
    "__version__",
    "ENGINE_VERSION",
    "REF_FORMAT_VERSION",
    # Port v3
    "port",
    "Consume",
    "compile_segment",
    "Segment",
    "Result",
    "Circuit",
    "CompiledDetectorSampler",
    "CompiledSampler",
    "DemWithReject",
    "PostselectFault",
    "TwirlDetectorSampler",
    "Diagnosis",
    "ExpectationInfo",
    "diagnose",
    "PostselectedLER",
    "postselected_logical_error_rate",
    "Reference",
    "Task",
    "collect",
    "scale_noise",
    "load_example",
    "example_path",
    "list_examples",
    "XtimCacheWarning",
    "XtimDemError",
    "XtimError",
    "XtimParseError",
    "XtimPerformanceWarning",
    "XtimRejectError",
    "XtimReferenceError",
    "XtimStimMissingError",
    "cache_dir",
    "compile_twirl_sampler",
    "compile_twirl_sampler_from_state",
    "bare_state_of",
    "input_state_of",
    "extract_frame",
    "fidelity_from_logicals",
]


def __dir__() -> list[str]:
    # Keep tab-completion / dir(xtim) to the public surface, hiding the submodules and
    # the `annotations` future-import binding.
    return list(__all__)
