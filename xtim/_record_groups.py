"""J-b1 P12: the layout-3 record-group reader (design §4 "port.py"), as Python taking a buffer object.

Duck-typed on a layout-3 ``BarrierBuffer`` (``record_keys``, ``record_group_ids``, ``outcomes``, ``plan_content``,
``dets``, ``decisions``, ``frames``, ``signs``): it never isinstance-checks an extension type, so the same code reads
the test API's twin now and the production buffer at F (plan r3.3).

A group is the set of shots with byte-identical layout-3 keys ([kind][len] content ‖ P_rec words ‖ y words): the
retained state's sufficient statistic, so every shot of a group carries the same state. σ is not part of the key
(design §4: σ retired); groups therefore split only by plan content, P_rec (frames included) and y.

Per group the reader exports:
  * group_plan_content: (kind, content bytes, 128-bit digest) for kind 0 and kind 2 alike, plus the S^a·CZ view
    (a wires, CZ pairs) when the content is kind 0 (``group_plan``);
  * group_outcomes: the outcomes u ‖ z of the group's first shot (identical across the group by construction);
  * group_prefix: the P_rec words (X then Z) — the per-group frame correction rows;
  * group_signs(paulis): signs() of the group's first shot.
Removed (design §4): group_sigmas, sig_bits, group_plans, group_plans_or_none, group_exact.
"""
from __future__ import annotations

import struct

import numpy as np


class RecordGroups:
    """Groups of a layout-3 barrier buffer (first-appearance order)."""

    def __init__(self, buffer):
        self._b = buffer
        self.group_id = np.asarray(buffer.record_group_ids(), np.int32)
        keys, counts = buffer.record_groups()
        self.group_keys = list(keys)
        self.group_counts = np.asarray(counts, np.int64)
        self.n_groups = len(self.group_keys)
        first = np.full(self.n_groups, -1, np.int64)
        seen = np.zeros(self.n_groups, bool)
        for i, g in enumerate(self.group_id.tolist()):
            if not seen[g]:
                seen[g] = True
                first[g] = i
        self.group_first_shot = first
        self._lay = buffer.record_layout()

    # ── per-group columns ────────────────────────────────────────────────────────────────────────
    def _first_rows(self, a):
        a = np.asarray(a)
        return a[self.group_first_shot] if a.ndim and a.shape[0] else a

    def group_dets(self) -> np.ndarray:
        """The packed detector row of each group's first shot (every shot of a group has the same y, but detectors
        include readout-flip coins, so this is the FIRST shot's row, not a group invariant)."""
        return self._first_rows(self._b.dets())

    def group_decisions(self) -> np.ndarray:
        return self._first_rows(self._b.decisions())

    def group_outcomes(self) -> list:
        return [np.asarray(self._b.outcomes(int(i))) for i in self.group_first_shot]

    def group_plan_content(self) -> list:
        out = []
        for i in self.group_first_shot:
            content, digest = self._b.plan_content(int(i))
            kind = content[0]
            (L,) = struct.unpack("<I", content[1:5])
            out.append((kind, bytes(content[5:5 + L]), bytes(digest)))
        return out

    def group_plan(self, g: int) -> dict:
        """The plan of group g: {kind, content, digest} and, for kind 0, its S^a·CZ view {a, cz}."""
        kind, content, digest = self.group_plan_content()[g]
        d = {"kind": kind, "content": content, "digest": digest}
        if kind == 0:
            n = int(self._lay["n"])
            d["a"] = [q for q in range(n) if content[q]]
            cz = []
            for j in range(n + 1, len(content) - 3, 4):
                cz.append((content[j] | (content[j + 1] << 8), content[j + 2] | (content[j + 3] << 8)))
            d["cz"] = cz
        return d

    def group_prefix(self) -> np.ndarray:
        """uint64[groups, 2 * PNW]: P_rec of each group (X words then Z words), from its layout-3 key."""
        W = int(self._lay["prefix_words"])
        out = np.zeros((self.n_groups, W), np.uint64)
        for g, k in enumerate(self.group_keys):
            (L,) = struct.unpack("<I", k[1:5])
            off = 5 + L
            out[g] = np.frombuffer(k[off:off + 8 * W], dtype="<u8")
        return out

    def group_signs(self, paulis) -> np.ndarray:
        """int8[groups, len(paulis)]: signs() (+1 / -1 definite, 0 otherwise) of each group's first shot."""
        s = np.asarray(self._b.signs(list(paulis)))
        return s[self.group_first_shot]
