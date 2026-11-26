"""Codegen gate (J-a r2.1; committed at J-b1 P0, plan §1.6 m8).
usage: python -I tools/codegen_eq.py <base.o> <tip.o> [--allow NAME ...] [--comdat <fam.o>] [--roots REGEX]
                                    [--disp] [--map OLD=NEW ...]

objdump -dr --no-show-raw-insn of every run_w<*> symbol (J-b2 J9; run_w<*, true> before J9; --roots overrides) and, transitively, of every
function those reference that is DEFINED in the same object (twirl_sampler.o), compared instruction by
instruction with symbol-relative offsets. Normalisation: instruction addresses -> offset in the function;
branch/call targets -> <symbol+off> only; relocations -> (type, resolved target) where section+offset
targets are resolved to the defining symbol (+delta) or, for read-only data, to the referenced bytes.
No base-name mapping unless --map is given (J-b2 J9: the pre-J-a run_w<N> -> run_w<N, false> rule is retired).
--comdat <fam.o>: every compared function ALSO defined in twirl_sampler_fam.o (inline Impl members emitted in both TUs,
COMDAT: the linker keeps one copy) must be instruction-identical between the two TUs, so the link order cannot pick a
different production body. Exit 0 iff no difference outside --allow names
(demangled prefix match). Prints one line per compared function and DIFF details.
--roots REGEX: the root functions (mangled names, re.search) instead of the production run_w<*, false> instances, e.g.
'run_wILi\\d+ELb1EEEbl$' to pin the FAM instances of twirl_sampler_fam.o (J-b1 P3 on). Exit 1 if no root matches.
J-b2 (plan r1 §4.0 / T2): --disp compares DISPLACEMENT-ONLY: memory displacements of every operand (member offsets
of Impl / CachedPlan that move when a data member is deleted) are masked, and intra-function branch targets are
compared by INSTRUCTION INDEX, not byte offset (an encoding-length change disp8 <-> disp32 moves every later offset).
Opcodes, registers, immediates, control flow and relocation targets are still compared. A speed statement, never a
correctness one. --map OLD=NEW (repeatable, regex on mangled names): base names are rewritten before matching (J9's
run_w<N, true> -> run_w<N>).
"""
import re, subprocess, sys
from collections import defaultdict

def sh(*a): return subprocess.run(a, check=True, capture_output=True, text=True).stdout

def load(obj):
    # sections: index -> name, and raw bytes of read-only data sections
    secs = {}
    for l in sh("readelf", "-SW", obj).splitlines():
        m = re.match(r"\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+\S+\s+(\S+)\s+(\S+)", l)
        if m: secs[m.group(2)] = (int(m.group(4), 16), int(m.group(5), 16))
    data = open(obj, "rb").read()
    def secbytes(name):
        off, size = secs[name]; return data[off:off + size]
    # symbols: per section, (value, size, name)
    bysec = defaultdict(list); defined = {}; labels = {}
    for l in sh("objdump", "-t", obj).splitlines():
        m = re.match(r"([0-9a-f]{16}) (.{7}) (\S+)\s+([0-9a-f]{16}) (?:\.hidden )?(\S+)$", l)
        if not m: continue
        val, flags, sec, size, name = int(m.group(1), 16), m.group(2), m.group(3), int(m.group(4), 16), m.group(5)
        if name.startswith(".L"): labels[name] = (sec, val)   # local constant labels (.LCn): TU-specific numbering
        if "F" in flags or "O" in flags:
            bysec[sec].append((val, size, name))
            if "F" in flags: defined[name] = (sec, val)
    return secs, secbytes, bysec, defined, labels

def resolve(target, ctx):
    secs, secbytes, bysec, defined, labels = ctx
    m = re.match(r"(\.[\w.$]+?)([+-]0x[0-9a-f]+)?$", target)
    if m and m.group(1) in labels:       # .LCn+add -> the label's CONTENT (+add): TU-specific numbering and section
        lsec, lval = labels[m.group(1)]; add = m.group(2) or ""
        if lsec.startswith(".rodata") and lsec in secs:
            b = secbytes(lsec); cst = re.search(r"\.cst(\d+)$", lsec)
            if ".str" in lsec:
                return "str:%r%s" % (b[lval:b.find(b"\0", lval)], add)
            return "ro:%s%s" % (b[lval:lval + (int(cst.group(1)) if cst else 16)].hex(), add)
        return "%s:label%s" % (lsec, add)
    if not m or m.group(1) not in secs:
        return target
    sec, off = m.group(1), int(m.group(2) or "0", 16)
    for val, size, name in bysec.get(sec, []):
        if val <= off < val + max(size, 1):
            return "%s+%d" % (name, off - val)
    if sec.startswith(".rodata"):
        b = secbytes(sec)
        if ".str" in sec:
            e = b.find(b"\0", off + 4 if off + 4 <= len(b) else off)
            return "str:%r" % b[max(off, 0):b.find(b"\0", max(off, 0))] if off >= 0 else "str?%r" % b[off + 4:e]
        return "ro:%s" % b[max(off, 0):max(off, 0) + 16].hex()
    return "%s+?" % sec

def disasm(obj, ctx):
    out = sh("objdump", "-dr", "--no-show-raw-insn", obj)
    funcs = {}; cur = None; base = 0
    for l in out.splitlines():
        m = re.match(r"([0-9a-f]+) <(.+)>:$", l)
        if m:
            cur = m.group(2); base = int(m.group(1), 16); funcs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s+([0-9a-f]+):\s+(R_\S+)\s+(\S+)$", l)
        if m:
            # the relocated field's displayed target is a placeholder (next address): the reloc names it
            j = len(funcs[cur]) - 1
            while j >= 0 and funcs[cur][j][0] != "ins": j -= 1
            if j >= 0:
                k, o, txt = funcs[cur][j]
                funcs[cur][j] = (k, o, re.sub(r"\s*(# )?[0-9a-f]* ?<[^>]*>", "", txt) + " <reloc>")
            tgt = m.group(3); t2 = re.sub(r"([+-]0x[0-9a-f]+)$", "", tgt); add = tgt[len(t2):]
            if t2 in ctx[0] or re.match(r"\.[\w.$]+$", t2):
                funcs[cur].append(("reloc", int(m.group(1), 16) - base, m.group(2), resolve(tgt, ctx)))
            else:
                funcs[cur].append(("reloc", int(m.group(1), 16) - base, m.group(2), t2 + add))
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+(.*)$", l)
        if m:
            ins = re.sub(r"\s+", " ", m.group(2).strip())
            ins = re.sub(r"\b[0-9a-f]+ <", "<", ins)             # absolute branch target -> <sym+off>
            ins = re.sub(r"<([^>+]+)\+0x([0-9a-f]+)>", lambda q: "<%s+%d>" % (q.group(1), int(q.group(2), 16)), ins)
            funcs[cur].append(("ins", int(m.group(1), 16) - base, ins))
    # aliases (a C1/C2 or D1/D2 pair at one address): objdump labels one name; the other shares its body
    defined = ctx[3]
    at = {}
    for nm, loc in defined.items():
        if nm in funcs: at.setdefault(loc, nm)
    for nm, loc in defined.items():
        if nm not in funcs and loc in at: funcs[nm] = funcs[at[loc]]
    return funcs

MAPS = []

def norm_name(n):
    for o, nw in MAPS:
        n = re.sub(o, nw, n)
    return n

def disp_norm(name, body):
    """--disp: memory displacements masked (an absent displacement is displacement 0); alignment padding (nop forms)
    dropped; self-branch targets by instruction index (encoding-length and padding neutral)."""
    def pad(t):
        return re.match(r"(?:(?:cs|ds|data16|rex\.\w*) )*(?:nop|xchg %ax,%ax)", t) is not None
    idx = {}
    k = 0
    for it in body:
        if it[0] == "ins":
            idx[it[1]] = k
            if not pad(it[2]):
                k += 1
    out = []
    for it in body:
        if it[0] == "ins":
            if pad(it[2]):
                continue
            t = re.sub(r"(?<![\w$])(-?0x[0-9a-f]+)?\(%", "D(%", it[2])
            t = re.sub(r"<%s\+(\d+)>" % re.escape(name), lambda q: "<self#%s>" % idx.get(int(q.group(1)), "?"), t)
            out.append(("ins", t))
        else:
            out.append(("reloc", it[2], it[3]))
    return out


def refs(body):
    r = set()
    for it in body:
        s = it[-1]
        for m in re.finditer(r"<([^>+]+)", s): r.add(m.group(1))
        if it[0] == "reloc": r.add(re.sub(r"([+-]0x[0-9a-f]+|\+\d+)$", "", s))
    return r

def demangle(names):
    return dict(zip(names, sh("c++filt", *names).splitlines())) if names else {}

def main():
    a = sys.argv[1:]; allow = []; fam_o = None; root_re = r"run_wILi\d+E(?:Lb1E)?EEbl$"; disp = False   # run_w<N> (J-b2 J9) or run_w<N, true> (J3..J8)
    if "--disp" in a:
        a.remove("--disp"); disp = True
    while "--map" in a:
        i = a.index("--map"); o, nw = a[i + 1].split("=", 1); MAPS.append((o, nw)); del a[i:i + 2]
    if "--roots" in a:
        i = a.index("--roots"); root_re = a[i + 1]; del a[i:i + 2]
    if "--comdat" in a:
        i = a.index("--comdat"); fam_o = a[i + 1]; del a[i:i + 2]
    while "--allow" in a:
        i = a.index("--allow"); allow.append(a[i + 1]); del a[i:i + 2]
    base_o, tip_o = a
    cb, ct = load(base_o), load(tip_o)
    fb = {norm_name(k): [(x[0], x[1], *map(norm_name, x[2:])) for x in v] for k, v in disasm(base_o, cb).items()}
    ft = disasm(tip_o, ct)
    defined_t = set(ct[3]); defined_b = {norm_name(n) for n in cb[3]}
    roots = [n for n in ft if re.search(root_re, n)]
    seen, todo = set(), list(roots)
    while todo:
        n = todo.pop()
        if n in seen: continue
        seen.add(n)
        for r in refs(ft.get(n, [])) | refs(fb.get(n, [])):
            if (r in defined_t or r in defined_b) and r not in seen: todo.append(r)
    if allow:                                  # the allowed functions are compared and listed too
        alln = sorted(set(ft) | set(fb)); da = demangle(alln)
        seen |= {n for n in alln if any(da[n].startswith(p) for p in allow)}
    dm = demangle(sorted(seen))
    ndiff = 0
    for n in sorted(seen, key=lambda x: dm.get(x, x)):
        d = dm.get(n, n)
        allowed = any(d.startswith(p) for p in allow)
        if n not in fb or n not in ft:
            st = "ONLY-IN-%s" % ("TIP" if n in ft else "BASE")
        else:
            st = "SAME" if fb[n] == ft[n] else "DIFF"
            if st == "DIFF" and disp and disp_norm(n, fb[n]) == disp_norm(n, ft[n]):
                st = "SAME-DISP"
        if st not in ("SAME", "SAME-DISP") and not allowed: ndiff += 1
        print("%-12s %s%s  (%d insns)" % (st, d, "  [allowed]" if allowed and st != "SAME" else "", len(ft.get(n, fb.get(n, [])))))
        if st == "DIFF":
            for i, (x, y) in enumerate(zip(fb[n], ft[n])):
                if x != y:
                    print("    first diff at item %d:\n      base: %s\n      tip:  %s" % (i, x, y)); break
            else:
                print("    length differs: base %d tip %d" % (len(fb[n]), len(ft[n])))
    ncom = nshared = 0
    if fam_o:
        # J-b1 (fixplan-coldstart review M4): only VAGUE-linkage symbols are shared by the linker (nm W/V/w/v in
        # BOTH objects); local clones (.constprop / .isra, binding t) are per-object and never linked across TUs, so
        # they are excluded by rule.
        def weak(o):
            out = set()
            for l in sh("nm", o).splitlines():
                f = l.split()
                if len(f) >= 3 and f[1] in ("W", "V", "w", "v"): out.add(f[2])
            return out
        wk = weak(fam_o) & weak(tip_o)
        ff = disasm(fam_o, load(fam_o))
        for n in sorted(seen, key=lambda x: dm.get(x, x)):
            if n in ff and n in ft and n in wk:
                nshared += 1
                print("COMDAT-SHARED %s" % dm.get(n, n))
                if ff[n] != ft[n]:
                    ncom += 1; print("COMDAT-DIFF  %s" % dm.get(n, n))
        print("COMDAT shared=%d DIFFS=%d" % (nshared, ncom))
    ndiff += ncom
    print("CODEGEN functions=%d roots=%d DIFFS=%d" % (len(seen), len(roots), ndiff))
    sys.exit(1 if ndiff or not roots else 0)

if __name__ == "__main__":
    main()
