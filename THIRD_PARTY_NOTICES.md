# Third-party files

xtim's own code and documentation are MIT-licensed (see `LICENSE`). The circuit files below come from third
parties and keep their original licenses. Each file's header names its source and lists the changes made for xtim.

## SOFT (Apache License 2.0)

Li, H. et al., "SOFT: Simulating quantum fault tolerance" (2025), https://github.com/haoliri0/SOFT, file
`magic_state_cultivation/circuits/circuit_d5_p0.005.stim`. Licensed under the Apache License, Version 2.0; a copy of
the license is in `LICENSES/Apache-2.0.txt`. Modified for xtim, as described in each file's header.

- `examples/color5_cultivation.stim`
- `xtim/_examples/color5_cultivation.stim`
- `tests/data/magna_color5_fid.stim`

## Magic state cultivation data (CC BY 4.0)

Gidney, C., Jones, C. and Shutty, N., data for "Magic state cultivation: growing T states as cheap as CNOT gates",
Zenodo record 13777072, https://zenodo.org/records/13777072. Licensed under the Creative Commons Attribution 4.0
International license, https://creativecommons.org/licenses/by/4.0/. Adapted for xtim, as described in each file's
header.

- `examples/cultivation_d3_faithful.stim`, `examples/cultivation_d3_rate.stim`
- `xtim/_examples/cultivation_d3_faithful.stim`, `xtim/_examples/cultivation_d3_rate.stim`
- `benchmarks/cultivation_d3_rate.stim`
- `tests/data/cultivation_d3_faithful.stim`, `tests/data/cultivation_d3_rate.stim`
- `examples/cultivation_d5.stim`, `xtim/_examples/cultivation_d5.stim`, `tests/data/cultivation_d5.stim`, transcribed
  from the record's `c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim` (same qubit
  layout) and modified for xtim: T/T_DAG gates in place of the S/S_DAG stand-ins, more detectors, output on fresh wires

## merlin benchmark generator

`tests/data/merlin_switch3_bt27_p0.stim` was exported from https://github.com/mark-koch/merlin (see its header).
It is test data in the source repository only and is not shipped in the wheel or the sdist.
