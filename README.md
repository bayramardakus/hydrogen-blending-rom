# Composition-Aware Reduced-Order Modelling of Real-Time Hydrogen Blending in Natural Gas Pipeline Networks

Code and data that reproduce every figure, table and number in the paper (International Journal of Hydrogen Energy, manuscript HE-D-26-11255). Nothing in the paper is hand-entered.

## Layout

- `work/` : all scripts and cached results. Start with `work/README.md`, which documents what each script produces.
- `scigrid/data/` : place the IGGIELGN v1.1.2 CSV files here. The data set is third-party licensed and is not redistributed.
- `ms/figs/` : figure output (override with the `PAPER_FIGS` environment variable).

## Data

The SciGRID_gas IGGIELGN data set (v1.1.2) is archived on Zenodo: https://doi.org/10.5281/zenodo.4767098

The sub-network extraction is deterministic: run against that release it returns a bit-identical sub-network description on repeated executions.

## Reproducing

Requires Python 3.10 or later with `numpy scipy matplotlib pandas networkx cvxpy CoolProp pandapipes` (CVXPY must have the CLARABEL solver).

Run `python run_all.py` inside `work/`, or run individual scripts as described in `work/README.md`.

## Citation

B. A. Kus, Composition-Aware Reduced-Order Modelling of Real-Time Hydrogen Blending in Natural Gas Pipeline Networks, under review at the International Journal of Hydrogen Energy.
