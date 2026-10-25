"""F-X: the exact-engine leaf set and the reachability gate (J-b1 plan r2.7 / r3.5; committed at P0).

Leaves = (out-of-line callees of the exact routes) MINUS (callees of the law / retention roots), computed on objects
built with -fno-inline (so every callee is a call edge), joined with the production .so's call graph for the edges
that cross translation units. The overlap (callees shared by an exact route and a law route) is reported, never
hidden. The four exact route functions themselves are always leaves (r3.5: forced noinline in the legacy instance).

usage:
  python -I tools/fx_leaves.py leaves <out.json> --so <_xtim.so> --obj <noinline.o> [--obj ...]
                                     [--exact-root REGEX ...] [--law-root REGEX ...] [--note TEXT]
  python -I tools/fx_leaves.py check <leaves.json> --so <_xtim.so> --root REGEX [--root ...]
                                     [--indirect-ok REGEX ...]
  python -I tools/fx_leaves.py absent <list.txt> --so <A.so> [--so ...] [--src DIR ...]
  python -I tools/fx_leaves.py dead <out.json> --so <_xtim.so> [--root REGEX ...] [--cut REGEX ...]
absent (J-b2, G-ABSENT): every non-comment line of <list.txt> is a regex (one deleted symbol, rodata string or Python
name). FAILS (exit 1) on any match in `nm -C` of a --so (GCC clone suffixes such as " [clone .cold]" are part of the
demangled name, so a pattern matches its clones), in the printable strings of a --so, or in a source file under a --src
dir (*.cpp *.hpp *.h *.py *.sh). A line "pattern  # src-ok" is checked in the binaries only.
dead (J-b2): over-approximating reachability on the linked .so: edges = every <sym> an instruction references (call, jmp,
lea, mov: address-taken functions are reached); roots = PyInit_* + every RELATIVE / GLOB_DAT / JUMP_SLOT target
(vtables, pointer tables) + --root; --cut removes functions from the graph (e.g. a deleted instance). Writes the qeccore
functions NOT reached (dead) and the closure size.
check: transitive closure over the WHOLE .so from every function matching a --root (direct calls, tail jumps, PLT),
FAILS (exit 1) if a leaf is reachable (the path is printed) or if a function in the closure makes an indirect call
and matches no --indirect-ok pattern (its resolved targets must be listed as extra --root). Patterns are re.search
over DEMANGLED names.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict, deque

EXACT_ROOTS = [r"TwirlRecordSampler::Impl::exact_ppr_shot\(", r"TwirlRecordSampler::Impl::exact_diag_shot\(",
               r"TwirlRecordSampler::Impl::exact_record_\(", r"TwirlRecordSampler::Impl::materialize_exact_shot\("]
LAW_ROOTS = [r"TwirlRecordSampler::Impl::run_w<\d+(?:, true)?>\(", r"TwirlRecordSampler::family_materialize",
             r"TwirlRecordSampler::family_signs", r"TwirlRecordSampler::family_record_",
             r"TwirlRecordSampler::Impl::fam_term_shot_\("]


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def demangle(names):
    names = sorted(names)
    if not names:
        return {}
    out = []
    for i in range(0, len(names), 2000):
        out += subprocess.run(["c++filt"], input="\n".join(names[i:i + 2000]) + "\n", check=True,
                              capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out))


def call_graph(path):
    """{function: (set of direct callees, number of indirect call/jmp sites)} from objdump -dr (mangled names).
    Relocated call targets (objects) are taken from the relocation; linked targets (.so) from the <sym> label."""
    edges, indirect = defaultdict(set), defaultdict(int)
    cur = None
    last_branch = False                      # the previous instruction is a call / jmp (its relocation names it)
    for line in sh("objdump", "-dr", "--no-show-raw-insn", path).splitlines():
        m = re.match(r"[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1)
            edges[cur]
            last_branch = False
            continue
        if cur is None:
            continue
        m = re.match(r"\s+[0-9a-f]+:\s+R_X86_64_(PLT32|PC32)\s+(\S+?)(?:[+-]0x[0-9a-f]+)?$", line)
        if m:
            if last_branch and not m.group(2).startswith("."):
                edges[cur].add(m.group(2))
            continue
        m = re.match(r"\s+[0-9a-f]+:\s+(call|jmp)\S*\s+(.*)$", line)
        last_branch = bool(m)
        if not m:
            continue
        arg = m.group(2)
        if "*" in arg.split("<")[0]:
            if "(%rip)" not in arg:          # call *%rax / *0x8(%rdx): an indirect target
                indirect[cur] += 1
            continue
        t = re.search(r"<([^>+@]+)(?:@plt)?(?:\+0x[0-9a-f]+)?>", arg)
        if t and t.group(1) != cur:
            edges[cur].add(t.group(1))
    return edges, indirect


def merged_graph(paths):
    E, I = defaultdict(set), defaultdict(int)
    for p in paths:
        e, i = call_graph(p)
        for k, v in e.items():
            E[k] |= v
        for k, v in i.items():
            I[k] = max(I[k], v)
    return E, I


def closure(E, roots, barriers=()):
    seen, parent, todo = set(roots), {r: None for r in roots}, deque(roots)
    while todo:
        n = todo.popleft()
        if n in barriers and n not in roots:
            continue
        for c in E.get(n, ()):
            if c not in seen:
                seen.add(c)
                parent[c] = n
                todo.append(c)
    return seen, parent


def qualified_name(d):
    """The qualified function name of a demangled signature: the text before the parameter list at template depth 0,
    without a return type (the last depth-0 space-separated token)."""
    depth, cut = 0, len(d)
    for i, ch in enumerate(d):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    head, depth, start = d[:cut], 0, 0
    for i, ch in enumerate(head):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == " " and depth == 0:
            start = i + 1
    return head[start:]


def engine_fn(d):
    """An engine function (a leaf candidate): a qeccore:: function that is not a constructor / destructor of a data
    type, not a lambda body and not a library template instance (std::, __gnu_cxx::)."""
    q = qualified_name(d)
    if not q.startswith("qeccore::"):
        return False
    parts = [x.split("<")[0] for x in re.split(r"::(?![^<]*>)", q)]
    return not (parts[-1].startswith("~") or (len(parts) > 1 and parts[-1] == parts[-2]))


def match(dm, pats):
    return {n for n, d in dm.items() if any(re.search(p, d) for p in pats)}


def _opts(a, keys):
    out = {k: [] for k in keys}
    while a:
        if a[0] not in out or len(a) < 2:
            raise SystemExit("bad argument " + a[0])
        out[a[0]].append(a[1])
        a = a[2:]
    return out


def leaves(argv):
    out = argv[0]
    o = _opts(argv[1:], ("--so", "--obj", "--exact-root", "--law-root", "--note"))
    E, _ = merged_graph(o["--obj"] + o["--so"])
    dm = demangle(set(E) | {c for v in E.values() for c in v})
    ex = match(dm, o["--exact-root"] or EXACT_ROOTS)
    law = match(dm, o["--law-root"] or LAW_ROOTS)
    if len(ex) < 4 or not law:
        raise SystemExit(f"roots not found: exact={sorted(dm[n] for n in ex)} law={len(law)}")
    ce, _ = closure(E, sorted(ex))
    cl, _ = closure(E, sorted(law), barriers=ex)
    body = set(E)                            # functions (a disassembled body somewhere); data symbols never are
    lv = sorted(n for n in (ce - cl) | ex if n in body and engine_fn(dm.get(n, n)))
    ov = sorted(n for n in ce & cl - ex if n in body and engine_fn(dm.get(n, n)))
    json.dump({"note": " ".join(o["--note"]), "leaves": lv, "overlap": ov, "exact_roots": sorted(ex), "law_roots": sorted(law),
               "leaves_demangled": [dm.get(n, n) for n in lv], "overlap_demangled": [dm.get(n, n) for n in ov]},
              open(out, "w"), indent=1)
    print(f"FX-LEAVES exact_roots={len(ex)} law_roots={len(law)} leaves={len(lv)} overlap={len(ov)}")
    return 0


def check(argv):
    L = json.load(open(argv[0]))
    o = _opts(argv[1:], ("--so", "--root", "--indirect-ok"))
    E, I = merged_graph(o["--so"])
    dm = demangle(set(E) | {c for v in E.values() for c in v})
    roots = sorted(match(dm, o["--root"]))
    if not roots:
        print("FX-CHECK no root matched: FAIL")
        return 1
    seen, parent = closure(E, roots)
    bad = 0
    for lf in L["leaves"]:
        if lf in seen:
            bad += 1
            path, n = [], lf
            while n is not None:
                path.append(dm.get(n, n))
                n = parent[n]
            print("REACHABLE LEAF:", " <- ".join(path[:12]))
    unres = sorted(n for n in seen if I.get(n) and not any(re.search(p, dm.get(n, n)) for p in o["--indirect-ok"]))
    for n in unres:
        print("UNRESOLVED INDIRECT:", dm.get(n, n), f"({I[n]} sites)")
    print(f"FX-CHECK roots={len(roots)} closure={len(seen)} reachable_leaves={bad} unresolved_indirect={len(unres)}")
    return 1 if bad or unres else 0


def absent(argv):
    pats = []
    for line in open(argv[0]):
        t = line.rstrip("\n")
        if not t.strip() or t.lstrip().startswith("#"):
            continue
        src_ok = t.endswith("# src-ok")
        pats.append((re.compile(t.split("  #")[0].strip()), src_ok))
    o = _opts(argv[1:], ("--so", "--src"))
    bad = 0
    for so in o["--so"]:
        syms = sh("nm", "-C", "--defined-only", so).splitlines()
        strs = subprocess.run(["strings", "-a", "-n", "6", so], check=True, capture_output=True, text=True).stdout.splitlines()
        for rx, _ in pats:
            for where, lines in (("nm", syms), ("strings", strs)):
                hit = next((x for x in lines if rx.search(x)), None)
                if hit:
                    bad += 1
                    print(f"PRESENT {where} {so.split('/')[-1]}: /{rx.pattern}/ -> {hit[:160]}")
    import os
    for d in o["--src"]:
        for dp, _, fs in os.walk(d):
            if "/build" in dp or "__pycache__" in dp:
                continue
            for f in fs:
                if not f.endswith((".cpp", ".hpp", ".h", ".py", ".sh")):
                    continue
                path = os.path.join(dp, f)
                if path.endswith("tools/fx_leaves.py"):
                    continue
                txt = open(path, errors="replace").read()
                for rx, src_ok in pats:
                    if src_ok:
                        continue
                    m = rx.search(txt)
                    if m:
                        bad += 1
                        ln = txt.count("\n", 0, m.start()) + 1
                        print(f"PRESENT src {path}:{ln}: /{rx.pattern}/")
    print(f"FX-ABSENT patterns={len(pats)} present={bad}")
    return 1 if bad else 0


def dead(argv):
    out = argv[0]
    o = _opts(argv[1:], ("--so", "--root", "--cut"))
    so = o["--so"][0]
    addr2fn = {}
    for line in sh("nm", "--defined-only", so).splitlines():
        p = line.split()
        if len(p) == 3 and p[1] in "tTwWiu":
            addr2fn.setdefault(int(p[0], 16), p[2])
    E = defaultdict(set)
    cur = None
    for line in sh("objdump", "-d", "--no-show-raw-insn", so).splitlines():
        m = re.match(r"[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1); E[cur]; continue
        if cur is None or ":" not in line:
            continue
        for t in re.findall(r"<([^>+@]+)(?:@plt)?(?:\+0x[0-9a-f]+)?>", line):
            if t != cur:
                E[cur].add(t)
    roots = {n for n in E if n.startswith("PyInit_")}
    for line in sh("readelf", "-rW", so).splitlines():
        if "R_X86_64_RELATIVE" in line:
            a = int(line.split()[-1], 16)
            if a in addr2fn:
                roots.add(addr2fn[a])
        m = re.search(r"R_X86_64_(64|GLOB_DAT|JUMP_SLOT)\s+[0-9a-f]+\s+(\S+)", line)
        if m:
            roots.add(m.group(2).split("@")[0])
    names = sorted(E)
    dm = demangle(set(names))
    roots |= {n for n in names if any(re.search(p, dm.get(n, n)) for p in o["--root"])}
    cut = {n for n in names if any(re.search(p, dm.get(n, n)) for p in o["--cut"])}
    roots -= cut
    seen, _ = closure({k: v - cut for k, v in E.items()}, sorted(roots))
    dd = sorted(dm.get(n, n) for n in names if n not in seen and "qeccore" in dm.get(n, n) and "@plt" not in n)
    json.dump({"so": so, "roots": len(roots), "closure": len(seen), "cut": sorted(dm.get(n, n) for n in cut),
               "dead": dd}, open(out, "w"), indent=0)
    print(f"FX-DEAD roots={len(roots)} closure={len(seen)} qeccore_dead={len(dd)}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "absent":
        sys.exit(absent(a[1:]))
    if a and a[0] == "dead":
        sys.exit(dead(a[1:]))
    if a and a[0] == "leaves":
        sys.exit(leaves(a[1:]))
    if a and a[0] == "check":
        sys.exit(check(a[1:]))
    raise SystemExit(__doc__)
