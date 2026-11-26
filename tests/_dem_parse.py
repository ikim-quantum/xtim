"""Tag-aware parsing of xtim DEM `error` lines (dem-erasure T4).

xtim emits the Pauli mechanisms as `error(p) D.. L..` and each Clifford fault's lines as
`error[clifford:f<id>](p) D.. L..` (stim's instruction-tag syntax). A parser that matches only
`error(` silently skips the Clifford records, so every test helper parses through here.
"""
import re

_ERR = re.compile(r"^error(?:\[([^\]]*)\])?\(([^)]*)\)\s*(.*)$")


def parse_error_line(line):
    """(p, components, tag) of one `error` line, or None for any other line. components: a list of
    target-name lists (one per `^`-separated component); tag: the `[...]` tag text or None."""
    m = _ERR.match(line.strip())
    if not m:
        return None
    comps = [c.split() for c in m.group(3).split("^")]
    return float(m.group(2)), comps, m.group(1)


def xor_compose(a, b):
    """Independent composition of two mechanisms on one signature: p (+) q = p + q - 2pq."""
    return a + b - 2 * a * b
