"""Link check (J-a r2.1; committed at J-b1 P0, plan §1.6 m8). usage: python -I tools/codegen_so.py <base _xtim.so> <tip _xtim.so> <tip twirl_sampler.o> <tip twirl_sampler_fam.o> [--allow NAME ...] [--base-ts <base twirl_sampler.o>]
Inline members emitted in several TUs are COMDAT: the linker keeps one copy, chosen by link order. So the linked .so is
the production artefact to gate. For every function defined in the tip's twirl_sampler.o or twirl_sampler_fam.o and
present in the base .so, the linked body must be identical base vs tip (addresses, RIP displacements and call targets
masked; opcodes, registers and immediates compared). Functions absent from the base .so are listed (NEW). A function
defined only in the FAM TU may differ (FAMDIFF, listed: the FAM instances are pinned by codegen_eq --roots); every
function defined in the production TU (twirl_sampler.o, COMDAT members included) must be identical. Exit 1 on any
production difference outside --allow (mangled-name prefix).
J-b2: --disp also masks every memory displacement (member offsets that move when a data member is deleted: the
DISPLACEMENT-ONLY class of plan r1 §4.0; a speed statement, never a correctness one). GONE entries for deleted
legacy functions are allowed by --allow (declared per commit)."""
import re, subprocess, sys
a = sys.argv[1:]; allow = []; base_ts = None; disp = "--disp" in a
if disp:
    a.remove("--disp")
if "--base-ts" in a:
    i = a.index("--base-ts"); base_ts = a[i + 1]; del a[i:i + 2]
while "--allow" in a:
    i = a.index("--allow"); allow.append(a[i + 1]); del a[i:i + 2]
bso, tso, ts, fam = a
def sh(*x): return subprocess.run(x, check=True, capture_output=True, text=True).stdout
def funcs(obj):
    out = {}
    for l in sh("objdump", "-t", obj).splitlines():
        m = re.match(r"([0-9a-f]{16}) (.{7}) (\S+)\s+([0-9a-f]{16})\s+(?:\.hidden\s+)?(\S+)$", l)
        if m and "F" in m.group(2) and int(m.group(4), 16) > 0: out[m.group(5)] = m.group(3)
    return out
def body(obj, sym):
    ins = []
    for l in sh("objdump", "-d", "--no-show-raw-insn", "--disassemble=" + sym, obj).splitlines():
        m = re.match(r"\s+[0-9a-f]+:\s+(.*)$", l)
        if not m: continue
        x = re.sub(r"\s+", " ", m.group(1).split("#")[0].strip())
        x = re.sub(r"\b[0-9a-f]+ <[^>]*>", "<T>", x)
        x = re.sub(r"-?0x[0-9a-f]+\(%rip\)", "D(%rip)", x)
        x = re.sub(r"^(callq?|jmpq?|j[a-z]+) [0-9a-f]+$", r"\1 <T>", x)
        if disp:
            if re.match(r"(?:(?:cs|ds|data16|rex\.\w*) )*(?:nop|xchg %ax,%ax)", x):
                continue                         # alignment padding
            x = re.sub(r"(?<![\w$])(-?0x[0-9a-f]+)?\(%", "D(%", x)
        ins.append(x)
    return ins
prod = set(funcs(ts))                  # defined in the production TU (incl. COMDAT members shared with the FAM TU)
names = sorted(prod | set(funcs(fam)))
fb, ft = funcs(bso), funcs(tso)
nsame = nnew = ndiff = nfam = 0
def base_name(n):   # the base's run_w<N> is the tip's run_w<N, false>
    return re.sub(r"run_wILi(\d+)ELb0EEEbl$", r"run_wILi\1EEEbl", n)
for n in names:
    if n not in ft: continue
    bn = n if n in fb else base_name(n)    # a post-J-a base already has the run_w<N, false> names
    if bn not in fb:
        nnew += 1; print("NEW   %s" % n); continue
    if body(bso, bn) == body(tso, n): nsame += 1
    elif n not in prod:                # FAM-TU-only code (J-b1: the FAM instances change; pinned by codegen_eq --roots)
        nfam += 1
        print("FAMDIFF  %s" % n)
    else:
        ok = any(n.startswith(p) for p in allow)
        ndiff += 0 if ok else 1
        print("DIFF  %s%s" % (n, "  [allowed]" if ok else ""))
# J-b1 (fixplan-coldstart review M4-ii): a function the BASE's production TU defined (--base-ts: the base
# twirl_sampler.o) that the base .so links and the tip .so does not (a renamed or removed production member) is a
# production difference, never a silent NEW.
ngone = 0
tip_names = set(ft) | {base_name(x) for x in ft}
base_prod = set(funcs(base_ts)) if base_ts else set()
for bn in sorted(fb):
    if bn in base_prod and bn not in tip_names:
        ok = any(bn.startswith(p) for p in allow)
        ngone += 0 if ok else 1
        print("GONE  %s%s" % (bn, "  [allowed]" if ok else ""))
ndiff += ngone
print("LINK compared=%d same=%d new=%d fam_only_diffs=%d gone=%d DIFFS=%d" % (nsame + ndiff - ngone + nfam, nsame, nnew,
                                                                              nfam, ngone, ndiff))
sys.exit(1 if ndiff else 0)
