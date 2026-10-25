#!/usr/bin/env bash
# J-b1 Q1: run the docs / examples checks that DEPEND ON F (the production flip).
#
# Use after F lands, on a tree whose xtim extension is built from F (or later), e.g.
#     PYTHON=python tools/check_docs_postF.sh
# from any directory. It never builds or installs anything: it imports xtim from the repo root
# (PYTHONPATH), so the extension must already be built in place.
#
# Checks:
#   1. the tree is post-F (ENGINE_VERSION == xtim-engine-13);
#   2. the post-F API facts the docs state (auto removed, default twirl, typed refusals, removed
#      keywords and methods, env levers not read);
#   3. the doc tests: tests/test_readme_snippets.py (strict xfail before F, must pass from F on)
#      and tests/test_tour.py;
#   4. every examples/*.py script;
#   5. (retired with the tutorial notebook);
#   6. a grep gate for stale pre-F text in the docs, and the count of TODO(F) / OPEN-1 markers
#      that must be resolved before the docs merge.
# Exit status: 0 only if 1-4 and 6 pass (6 fails on stale text; TODO(F) markers are reported, and fail
# only with STRICT_TODO=1).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
fail=0
step() { printf '\n== %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

step "1. post-F tree"
ver="$("$PY" -c 'import xtim; print(xtim.ENGINE_VERSION)')" || { echo "cannot import xtim"; exit 2; }
echo "ENGINE_VERSION = $ver"
if [ "$ver" != "xtim-engine-13" ]; then
    echo "not a post-F tree (want xtim-engine-13): these checks depend on F; stopping."
    exit 2
fi

step "2. post-F API facts stated by the docs"
"$PY" - <<'EOF' || fail=1
import os, re, warnings
import xtim
warnings.simplefilter("ignore")
errs = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        errs.append(what)


def guarded(fn):
    try:
        fn()
    except Exception as e:  # a crash inside a block is a failed check, not a script abort
        check(False, f"{fn.__name__} raised {type(e).__name__}: {str(e)[:120]}")


c = xtim.load_example("distillation_15_1_3")


def auto_removed():
    try:
        c.compile_detector_sampler(engine="auto")
        check(False, 'engine="auto" raises ValueError')
    except ValueError as e:
        check("removed" in str(e) and "exact" in str(e), 'engine="auto" raises ValueError naming the removal')


def default_twirl():
    s = c.compile_detector_sampler(seed=0)
    check(type(s).__name__ == "TwirlDetectorSampler", "default engine is twirl")
    rep = s.engine_report()
    check(rep.get("engine") == "twirl" and rep.get("reason") == "default", 'engine_report() == twirl / "default"')
    for kw, cls in (("return_expectations", "EXPECTATIONS_NOT_SERVED"),
                    ("return_measurements", "RAW_RECORDS_NOT_SERVED")):
        try:
            s.sample(8, **{kw: True})
            check(False, f"{kw} refused")
        except xtim.XtimTwirlRefusal as e:
            check(e.twirl_class is getattr(xtim.TwirlClass, cls), f"{kw} -> XtimTwirlRefusal({cls})")
            m = str(e)
            if kw == "return_expectations":
                check("engine='exact'" in m or 'engine="exact"' in m, "the message names engine='exact'")
                check("sample_barrier" in m and "RecordFidelity" in m, "the message names sample_barrier + RecordFidelity")
            else:
                check("compile_sampler" in m, "the message names compile_sampler")


def explicit_exact():
    e = c.compile_detector_sampler(seed=0, engine="exact")
    _d, x = e.sample(8, return_expectations=True)
    check(x.shape[0] == 8, 'explicit engine="exact" serves return_expectations')
    check(e.engine_report().get("reason") == "requested", 'engine_report()["reason"] == "requested" on explicit exact')


def typed_refusals():
    try:
        xtim.Circuit("RX 0\nRX 1\nT 1\nCH 0 1\nM 0 1").compile_detector_sampler()
        check(False, "unaligned T->CH refused at compile")
    except xtim.XtimTwirlRefusal as e:
        check(e.twirl_class is xtim.TwirlClass.OUTSIDE_T_DEPTH_1, "unaligned T->CH -> OUTSIDE_T_DEPTH_1 at compile")
    want = {"OUTSIDE_T_DEPTH_1", "NONCOMMUTING_MAGIC", "LAW_LIMITATION_PROPAGATION", "LAW_LIMITATION_CARRIED_AXIS",
            "EXPECTATIONS_NOT_SERVED", "RAW_RECORDS_NOT_SERVED", "CORRECTION_NOT_SERVED"}
    check({m.name for m in xtim.TwirlClass} == want, "TwirlClass members match the docs' table")


def removed_surfaces():
    try:
        xtim.compile_twirl_sampler(c.text, skip_refused_observables=True)
        check(False, "skip_refused_observables= -> TypeError")
    except TypeError:
        check(True, "skip_refused_observables= -> TypeError")
    tw = xtim.compile_twirl_sampler(c.text)
    check(not hasattr(tw, "channel_report"), "channel_report() removed")
    check(hasattr(tw, "record_report"), "record_report() present")
    check(hasattr(xtim.RecordFidelity, "evaluate") and not hasattr(xtim.RecordFidelity, "fidelities"),
          "RecordFidelity.evaluate present, fidelities removed")
    src = "".join(open(os.path.join("xtim", f)).read() for f in os.listdir("xtim") if f.endswith(".py"))
    for lever in ("QEC_NO_TWIRL", "QEC_TW_SELFCHECK"):
        check(re.search(r"environ[^\n]*%s" % lever, src) is None, f"{lever} is not read by xtim/*.py")


for fn in (auto_removed, default_twirl, explicit_exact, typed_refusals, removed_surfaces):
    guarded(fn)
raise SystemExit(1 if errs else 0)
EOF

step "3. doc tests"
"$PY" -m pytest -q -p no:cacheprovider tests/test_readme_snippets.py tests/test_tour.py || bad "doc tests"

step "4. example scripts (run from a scratch directory: some write plots to the cwd)"
EXDIR="$(mktemp -d)"; trap 'rm -rf "$EXDIR"' EXIT
for f in examples/*.py; do
    printf -- '-- %s\n' "$f"
    (cd "$EXDIR" && MPLBACKEND=Agg "$PY" "$ROOT/$f" > /dev/null) || bad "$f"
done

step "6. stale pre-F text in docs / examples / the docstring-bearing modules"
# (a hit in module CODE means F did not remove that pre-F path: report it to the engine worker)
DOCS="README.md docs examples xtim/__init__.py xtim/circuit.py xtim/twirl.py xtim/collect.py xtim/ler.py xtim/errors.py"
DOCS="$DOCS xtim/record_fidelity.py xtim/port.py"
stale="$(grep -rnIE 'engine="auto"|engine=.auto.\)|\.fidelities\(|skip_refused_observables=True|channel_report\(\)\[|No shot is ever dropped|no shot is ever dropped|_exact_thunk|QEC_NO_TWIRL=1' \
    $DOCS --include='*.md' --include='*.py' --include='*.ipynb' 2>/dev/null \
    | grep -v '^docs/repin/' | grep -vE 'removed|REMOVED|replaces|renamed|TODO\(F\)|OPEN-1' || true)"
if [ -n "$stale" ]; then printf '%s\n' "$stale"; bad "stale pre-F text (above)"; else echo "ok   no stale pre-F text"; fi
todo="$(grep -rnIE 'TODO\(F\)|OPEN-1' README.md docs examples xtim CHANGELOG.md 2>/dev/null | grep -v '^docs/repin/' || true)"
ntodo="$(printf '%s' "$todo" | grep -c . || true)"
echo "TODO(F) / OPEN-1 markers left: $ntodo (resolve before the docs merge)"
[ "$ntodo" -gt 0 ] && printf '%s\n' "$todo" | sed 's/^/   /' | cut -c1-160
if [ "${STRICT_TODO:-0}" = 1 ] && [ "$ntodo" -gt 0 ]; then bad "TODO(F) markers left (STRICT_TODO=1)"; fi

printf '\n'
if [ "$fail" = 0 ]; then echo "check_docs_postF: PASS"; else echo "check_docs_postF: FAIL"; fi
exit "$fail"
