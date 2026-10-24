# xtim examples

Runnable scripts (`*.py`) and the bundled circuits (`*.stim`). The circuits also ship inside the wheel:
`xtim.load_example(name)` loads one, `xtim.list_examples()` lists them. Every bundled circuit declares its output
register (`OUTPUT_QUBITS`) with every frame closed, so `xtim.RecordFidelity` scores it directly.

Each script runs standalone (`python examples/<script>`), prints its result, and ends by checking that the
noiseless fidelity is exactly 1 on every shot.

| script | circuit | what it shows |
|---|---|---|
| `01_quickstart_distillation.py` | `distillation_15_1_3` | sample, post-select, score; 1-F of the accepted output vs p (the ~35 p^3 floor) |
| `02_cultivation_d3.py` | `cultivation_d3_faithful`, `cultivation_d3_rate` | cultivation: acceptance, F of the accepted shots vs all shots; the Clifford twin as a control |
| `03_color5_postselected.py` | `color5_cultivation` | d=5 colour-code cultivation (SOFT, Apache-2.0): post-selected F |
| `04_h_cultivation.py` | `ch_cultivation` | H-state cultivation with controlled-Hadamard checks |
| `05_ccz_cube.py` | `cube_ccz` | [[8,3,2]] transversal CCZ: a 3-qubit magic state; T-gate noise added by the script |
| `06_code_switching.py` | `code_switching_faithful`, `code_switching_rate` | qRM to Steane code switching with a Pauli-feedback frame |
| `07_refusals.py` | small inline circuits | the named refusals and how to fix each (incl. closing a teleportation frame) |
| `08_bring_your_own.py` | inline | declare the output, close the frame, score |
| `09_decoded_fidelity.py` | inline | a decoder's predicted observable flips as a Pauli correction on the output (needs `[decode]`) |

Bundled circuits:

| name | protocol |
|---|---|
| `distillation_15_1_3` | [[15,1,3]] quantum Reed-Muller code, transversal T, X-stabilizer detectors (15-to-1 distillation) |
| `cultivation_d3_faithful` / `_rate` | d=3 magic-state cultivation (T) / its Clifford twin (S); output moved onto fresh wires |
| `cultivation_d5` | d=5 magic-state cultivation; output moved onto fresh wires |
| `color5_cultivation` | SOFT's d=5 colour-code cultivation (Apache-2.0), frame closed, output on fresh wires |
| `ch_cultivation` | H-eigenstate cultivation by controlled-Hadamard checks |
| `cube_ccz` | [[8,3,2]] cube code, transversal CCZ (noiseless) |
| `code_switching_faithful` / `_rate` | qRM to Steane code switching (arXiv:2410.07327) / its count-matched twin |
| `miniature_oracle` | a T state teleported into a 3-qubit repetition code, frame closed by `CZ rec[-3] 0` |
