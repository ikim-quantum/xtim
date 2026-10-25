"""S8 pin manifest: every signed pin literal in tests/ (a quoted hex digest of >= 16 characters, with the source line
that carries its tuple fields), as sorted JSON. Diff two trees' manifests to show which pins a stage moves.

usage: python -I tools/pin_manifest.py <tree root> [> manifest.json]
"""
import json
import os
import re
import sys

root = sys.argv[1]
PAT = re.compile(r"""["']([0-9a-f]{16,64})["']""")
out = []
for dp, _dn, fn in os.walk(os.path.join(root, "tests")):
    for f in sorted(fn):
        if not f.endswith((".py", ".json")):
            continue
        p = os.path.join(dp, f)
        for ln, line in enumerate(open(p, errors="replace"), 1):
            for m in PAT.finditer(line):
                out.append(dict(file=os.path.relpath(p, root), line=ln, pin=m.group(1), text=line.strip()[:240]))
out.sort(key=lambda d: (d["file"], d["line"], d["pin"]))
json.dump(out, sys.stdout, indent=1)
print()
