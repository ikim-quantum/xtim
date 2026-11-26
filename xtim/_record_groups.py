"""The layout-3 record-group reader (used by xtim.port), as Python taking a buffer object.

Duck-typed on a layout-3 ``BarrierBuffer`` (``record_groups``, ``record_group_ids``, the batched per-row readers
``_record_group_first`` / ``_outcomes_at`` / ``_plan_content_at`` / ``_prefix_at`` / ``_signs_at``, ``dets``,
``decisions``, ``record_layout``): it never isinstance-checks an extension type, so the same code reads the test API's
twin and the production buffer.

A group is the set of shots with byte-identical layout-3 keys ([kind][len] content ‖ P_rec words ‖ y words): the
retained state's sufficient statistic, so every shot of a group carries the same state. σ is not part of the key
(σ retired in 3.2.0); groups therefore split only by plan content, P_rec (frames included) and y.

Per group the reader exports:
  * group_plan_content: (kind, content bytes, 128-bit digest) for kind 0 and kind 2 alike, plus the S^a·CZ view
    (a wires, CZ pairs) when the content is kind 0 (``group_plan``);
  * group_outcomes: the outcomes u ‖ z of the group's first shot (identical across the group by construction);
  * group_prefix: the P_rec words (X then Z) — the per-group frame correction rows;
  * group_signs(paulis): signs() of the group's first shot.
Removed in 3.2.0: group_sigmas, sig_bits, group_plans, group_plans_or_none, group_exact.

Every per-group reader is ONE batched engine call over the groups' first rows, built on first use and kept (the
buffer is immutable); group_signs evaluates the first rows only (signs() is a per-row function).
"""
from __future__ import annotations

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
        self.group_first_shot = np.asarray(buffer._record_group_first(), np.int64)
        self._lay = buffer.record_layout()
        self._outcomes = None
        self._plan_content = None
        self._prefix = None

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
        if self._outcomes is None:
            self._outcomes = list(self._b._outcomes_at(self.group_first_shot))
        return self._outcomes

    def group_plan_content(self) -> list:
        if self._plan_content is None:
            self._plan_content = list(self._b._plan_content_at(self.group_first_shot))
        return self._plan_content

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
        """uint64[groups, 2 * PNW]: P_rec of each group (X words then Z words), the P_rec words of its layout-3 key."""
        if self._prefix is None:
            self._prefix = np.asarray(self._b._prefix_at(self.group_first_shot), np.uint64)
        return self._prefix

    def group_signs(self, paulis) -> np.ndarray:
        """int8[groups, len(paulis)]: signs() (+1 / -1 definite, 0 otherwise) of each group's first shot."""
        return np.asarray(self._b._signs_at(list(paulis), self.group_first_shot))
