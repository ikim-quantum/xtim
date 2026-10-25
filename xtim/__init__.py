"""xtim — a Stim-superset simulator for magic-state preparation.

    import xtim
    sampler = xtim.compile_twirl_sampler(text)          # the circuit declares OUTPUT_QUBITS, frames closed
    buf = sampler.sample_barrier(10**6, seed=7)          # each shot keeps its exact output state
    dets = buf.dets()                                    # Stim b8-packed detection events
    F = xtim.RecordFidelity(sampler).evaluate(buf)       # exact per-shot fidelity of the output register

    dets, obs = sampler.sample(10**6, seed=7)            # detectors / observables only
    dem = xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True)   # a stim.DetectorErrorModel

A circuit or request outside the model is refused by name (``XtimTwirlRefusal``, ``XtimFidelityRefusal``);
nothing is silently approximated. See README.md.

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
    __version__ = "3.2.2+dev"
finally:
    globals().pop("_v", None)  # don't leak the import into the public namespace

# Where the invisible reference cache lives; None -> .xtim_cache/ under the cwd.
cache_dir: "str | None" = None

from ._xtim import ENGINE_VERSION, REF_FORMAT_VERSION  # noqa: E402
from ._xtim import live_state_bytes, state_budget_bytes  # noqa: E402
from .circuit import (  # noqa: E402
    Circuit,
    CompiledDetectorSampler,
    CompiledSampler,
    DemWithReject,
    PostselectFault,
    TwirlDetectorSampler,
)
from .collect import Task, collect, scale_noise  # noqa: E402
from .ler import PostselectedLER, postselected_logical_error_rate  # noqa: E402
from .examples import example_path, list_examples, load_example  # noqa: E402
from .errors import (  # noqa: E402
    XtimCacheWarning,
    XtimCapacityError,
    XtimDemError,
    XtimError,
    XtimInvariantError,
    XtimParseError,
    XtimPerformanceWarning,
    XtimVerificationWarning,
    XtimCorrectnessWarning,
    XtimDemWarning,
    XtimWarning,
    XtimRejectError,
    XtimReferenceError,
    XtimStimMissingError,
    XtimTwirlRefusal,
    XtimFidelityRefusal,
    TwirlClass,
)
from .reference import Reference  # noqa: E402
from .fidelity_helpers import extract_frame  # noqa: E402
# The state route (the default output scoring; unreleased): exact per-shot fidelity of a retained
# OUTPUT_QUBITS state with a target, in the bare state's shared stabilizer frame. The module
# path xtim.record_fidelity keeps working; these are the discoverable names.
from .record_fidelity import RecordFidelity, correction_from_observables  # noqa: E402
# Port v3: declared-consumption segment surface (facade over existing samplers).
# Import as xtim.port and expose the two nouns + two verbs at the top level.
from . import port  # noqa: E402
from .forced import ExactBranches, ExactBranchSimulator, ExpandedLawTooLarge  # noqa: E402
from .port import Consume, compile as compile_segment, Segment, Result  # noqa: E402
# Engine-level twirl surface (the record/barrier/decision sampler generation +
# port-contract state objects). This is the public API consumed by adaptq's
# M2b/M3 validation suite; the orchestration layer (feedback/stage_api/tier/...)
# lives in the monorepo and is deliberately NOT part of the standalone package.
from .twirl import (  # noqa: E402
    compile_twirl_sampler,
    compile_twirl_sampler_from_ideal_state,
    bare_state_of,
    input_state_of,
    # Private aliases, re-exported deliberately and NOT in __all__: they are
    # the pre-2.x spellings, kept so existing notebooks importing the
    # underscore names keep working.  Unused inside this repo by design --
    # suppressed rather than deleted, since removing a name from a PUBLISHED
    # package is a breaking change and 3.0.0 has already shipped.
    _bare_state_of,  # noqa: F401
    _input_state_of,  # noqa: F401
)

__all__ = [
    "ExactBranches",
    "ExactBranchSimulator",
    "ExpandedLawTooLarge",
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
    "XtimCapacityError",
    "XtimDemError",
    "XtimError",
    "XtimInvariantError",
    "XtimParseError",
    "XtimPerformanceWarning",
    "XtimVerificationWarning",
    "XtimCorrectnessWarning",
    "XtimDemWarning",
    "XtimWarning",
    "XtimRejectError",
    "XtimReferenceError",
    "XtimStimMissingError",
    "XtimTwirlRefusal",
    "XtimFidelityRefusal",
    "TwirlClass",
    "cache_dir",
    "live_state_bytes",
    "state_budget_bytes",
    "compile_twirl_sampler",
    "compile_twirl_sampler_from_ideal_state",
    "bare_state_of",
    "input_state_of",
    "extract_frame",
    # state route
    "RecordFidelity",
    "correction_from_observables",
]


def __dir__() -> list[str]:
    # Keep tab-completion / dir(xtim) to the public surface, hiding the submodules and
    # the `annotations` future-import binding.
    return list(__all__)
