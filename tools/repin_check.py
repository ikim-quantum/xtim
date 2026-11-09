"""Re-pin completeness check (J-b1 plan r2.9 MINOR-7; committed at P0).

Every hunk of `git diff <base>..<tip> -- <paths>` must map to a row of the re-pin manifest JSON. A row is
  {"file": "tests/test_x.py", "old": "<text in a removed line>", "new": "<text in an added line>",
   "mechanism": "M-...", "hunk": "<optional substring of the @@ header line>"}
and maps a hunk when the file matches and every given field matches ("old" in some removed line, "new" in some
added line, "hunk" in the header). An unmapped hunk FAILS (exit 1); rows that map nothing are listed (stale rows).

usage: python -I tools/repin_check.py <repo> <base> <tip> <manifest.json> [path ...]
       (default paths: tests/ README.md benchmarks/)
"""
import json
import subprocess
import sys


def hunks(diff_text):
    """[(file, header, removed lines, added lines)] of a unified diff."""
    out, f, cur = [], None, None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            f, cur = None, None
        elif line.startswith("+++ "):
            f = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            cur = [f, line, [], []]
            out.append(cur)
        elif cur is not None and line.startswith("-"):
            cur[2].append(line[1:])
        elif cur is not None and line.startswith("+"):
            cur[3].append(line[1:])
    return [tuple(h) for h in out]


def maps(row, h):
    f, header, rem, add = h
    if row.get("file") != f:
        return False
    if "hunk" in row and row["hunk"] not in header:
        return False
    if "old" in row and not any(row["old"] in x for x in rem):
        return False
    if "new" in row and not any(row["new"] in x for x in add):
        return False
    return any(k in row for k in ("hunk", "old", "new"))


def check(diff_text, rows):
    H = hunks(diff_text)
    used = set()
    unmapped = []
    for h in H:
        hit = [i for i, r in enumerate(rows) if maps(r, h)]
        if not hit:
            unmapped.append(h)
        used |= set(hit)
    return H, unmapped, [r for i, r in enumerate(rows) if i not in used]


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    repo, base, tip, man = argv[:4]
    paths = argv[4:] or ["tests/", "README.md", "benchmarks/"]
    d = subprocess.run(["git", "-C", repo, "diff", "-U0", f"{base}..{tip}", "--", *paths], check=True,
                       capture_output=True, text=True).stdout
    rows = json.load(open(man))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    H, un, stale = check(d, rows)
    for f, header, rem, add in un:
        print("UNMAPPED", f, header, "| -", (rem[:1] or [""])[0][:80], "| +", (add[:1] or [""])[0][:80])
    for r in stale:
        print("STALE ROW", json.dumps(r)[:160])
    print(f"REPIN hunks={len(H)} unmapped={len(un)} stale_rows={len(stale)}")
    return 1 if un else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
