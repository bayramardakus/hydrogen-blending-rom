# Composition-Aware Reduced-Order Modelling of Real-Time Hydrogen Blending

## Reproduction package

Every figure, table and number in the article is produced by the scripts in
`work/`. Nothing is hand-entered.

---

## 1. Requirements

```
python >= 3.10
numpy scipy matplotlib pandas networkx cvxpy CoolProp pandapipes
```

```bash
pip install numpy scipy matplotlib pandas networkx cvxpy CoolProp pandapipes
```

CVXPY must have the CLARABEL solver available (it ships with CVXPY 1.4 and
later).

Every study is seeded, and the cached `.npy` files in `work/` are the results
the article quotes. `work/verify_r2.py` checks the article against them; give it
the path to the manuscript source as its only argument.

One quantity is an exception, and it is worth stating plainly. In
`structural_analysis.py`, the `legacy` branch of `S2_observer` reconstructs the
badly tuned filter that the scaled formulation replaces. That filter has an
infinity norm of exactly one: it passes the raw measurement through and
differentiates it, so its output is ill-conditioned by construction and its
magnitude moves by a few per cent with the LAPACK build underneath SciPy. The
cached value is 3556 kg/s; another environment may return, for instance,
3415 kg/s. Nothing in the article turns on the exact figure. The two statements
it supports, that the mis-tuned filter is three orders of magnitude worse than
the physical tuning and that its gain norm is one, hold either way, and both are
checked by `verify_r2.py`.

## 2. Data

The SciGRID_gas IGGIELGN data set (v1.1.2) is **not** redistributed here.
Download it from Zenodo and unpack it so that the CSV files sit in
`scigrid/data/`:

```
https://doi.org/10.5281/zenodo.4767098
```

```
hydrogen-blending-rom/
  scigrid/data/IGGIELGN_PipeSegments.csv
  scigrid/data/IGGIELGN_Nodes.csv
  ...
  work/              <- the scripts and the cached results
  ms/figs/           <- figure output
```

The manuscript source is not included here; the article itself is the published
record. Scripts that need it take its path as an argument.

## 3. Reproducing everything

```bash
cd work
python run_all.py            # about 45 min on one core
```

or step by step, in this order, since later steps consume earlier `.npy`
outputs:

| # | Script | Produces | Article |
|---|--------|----------|---------|
| 1 | `extract_subnet.py` | `scigrid_subnet.json` | §5.1 |
| 2 | `i8_eos.py` | GERG-2008 comparison | §6.1, Table 4 |
| 3 | `advection_benchmark.py` | `advection_convergence.npy` | §6.2, Fig. 7a,b |
| 4 | `structural_analysis.py` | `structural_analysis.npy` | §6.2, §6.9, Figs. 7c, 18 |
| 5 | `realnet_pandapipes.py` | `realnet_pandapipes.npy` | §6.3, Fig. 8 |
| 6 | `realnet_reduction.py` | `realnet_reduction.npy` | §6.4 |
| 7 | `realnet_pressure.py` | `realnet_pressure_results.npy` | §6.8, Fig. 16 |
| 8 | `rerun_blend_adv.py` | `blend_adv_results.npy` | §6.7, Fig. 14 |
| 9 | `realnet_blend.py` | `realnet_blend_results.npy` | §6.8, Fig. 15 |
| 10 | `energy_accounting.py` | `energy_accounting.npy` | §6.11 |
| 11 | `verify.py` | `sweep.npy` | §6.4, Table 5, Fig. 10a |
| 12 | `parameter_studies.py` | `backoff_study.npy`, `wobbe_variant.npy`, `surge_sweep.npy` | §4.3, §6.8, §6.10, Fig. 17 |
| 13 | `robustness_studies.py` | `robustness_studies.npy` | §6.12, Fig. 20 |
| 14 | `coupling_reversal_uptake.py` | `coupling_reversal_uptake.npy` | §6.2, §6.8, §6.14 |
| 15 | `fig_parity.py` | the pandapipes parity figure | Fig. 8 |
| 16 | `make_new_figures.py` | the second group of figures | Figs. 7, 14 to 18 |
| 17 | `metrics_r2.py` | RMSE and R² added to `sweep.npy` | §6.4, Table 5 |
| 18 | `realnet_r2.py` | RMSE and R² added to `realnet_reduction.npy` | §6.4 |
| 19 | `uncertainty_budget.py` | `uncertainty_budget.npy` | §6.12, Table 6, Fig. 19 |
| 20 | `fig_uncertainty.py` | the error-analysis figure | Fig. 19 |
| 21 | `verify_r2.py <manuscript.tex>` | checks every number in the article against its result file | all of §6 |

Set `PAPER_FIGS` to redirect figure output; the default is `../ms/figs`.

## 4. Determinism

`extract_subnet.py` returns a **bit-identical** `scigrid_subnet.json` on
repeated runs against the archived data set:

```bash
for i in 1 2 3; do python extract_subnet.py > /dev/null; md5sum scigrid_subnet.json; done
```

Graph traversal in NetworkX is not order-stable, so two things are forced:

* node indices are assigned by **sorting on the SciGRID_gas identifier**;
* each pipe is written with a **canonical endpoint orientation** (`a < b`),
  which fixes the sign convention of the incidence matrix and therefore the
  reported sign of every pipe flow.

Without them a re-run reproduces the same topology but relabels 18 of the 22
nodes. For the same reason **every disturbance scenario is anchored by a rule on
a physical property**, never by a node number:

* `realnet.major_offtakes(net, k)`: the k largest withdrawals;
* `realnet.weakest_nodes(net, P0, k)`: the k nodes with the smallest nominal
  margin to the delivery floor;
* `realnet.far_nodes(net, k)`: the k nodes furthest from the supply.

## 5. File map

**Model**

- `gas_properties.py`: NG/H₂ mixture properties, Colebrook-White friction
- `network.py`: didactic network; scaled steady solve; RLC reduction; 1-D reference
- `realnet.py`: the real SciGRID_gas sub-network; rule-based scenario node selection
- `extract_subnet.py`: deterministic sub-network extraction from IGGIELGN
- `composition.py`, `composition_adv.py`: hydrogen transport (1-cell, M-cell)
- `transport_fir.py`: per-pipe and CFL-matched upwind; impulse-response form
- `estimator.py`: state scaling; physically-tuned steady-state Kalman filter

**Studies**

- `i8_eos.py`: GERG-2008 comparison
- `advection_benchmark.py`: transport discretisation convergence
- `structural_analysis.py`: controllability and observability, estimator
  robustness, frozen-flow envelope, roughness and temperature sensitivity
- `robustness_studies.py`: sparse pressure telemetry; green-hydrogen forecast error
- `coupling_reversal_uptake.py`: magnitude of the neglected two-way coupling;
  the demand amplitude at which a loop chord reverses on the real network;
  attribution of the green-hydrogen uptake figure
- `parameter_studies.py`: back-off sweep, Wobbe-band variant,
  contingency-magnitude sweep
- `sensitivity_analysis.py`: standalone sensitivity package, a superset of the
  structural studies above
- `realnet_pandapipes.py`, `pandapipes_validation.py`: code-to-code verification
- `realnet_reduction.py`, `accuracy_at_scale.py`, `verify.py`: reduction accuracy
- `scalability.py`: cost versus network size
- `grid_convergence.py`: grid convergence of the real-network pressure reference
- `ensemble_study.py`, `ci_ensemble.py`: the twenty-realisation ensemble and its
  confidence intervals
- `uncertainty_budget.py`: the consolidated uncertainty budget
- `metrics_r2.py`, `realnet_r2.py`: RMSE and R² for the two reduction studies
- `experiment_validation.py`: reduced model against the 1-D reference at 5 and
  25 vol% hydrogen

**Control**

- `realnet_pressure.py`: Case Study 3 pressure regulation
- `realnet_blend.py`: Case Study 3 two-point blend scheduling
- `rerun_blend_adv.py`: didactic blend scheduling
- `mpc.py`, `mpc_analysis.py`, `blend_mpc.py`: didactic controller studies
- `energy_accounting.py`: energy balance

**Figures and checks**

- `make_figures.py`, `make_new_figures.py`, `fig_*.py`: the article's figures
- `figstyle.py`: the shared figure style
- `regen_figs.py`: redraw every figure from the cached results, re-simulating
  nothing
- `verify_r2.py`: the verification pass over the article's numbers

## 6. Known scope limits

Stated in §6.13 and Table 5 of the article, and repeated here so that the code is
not read as claiming more than it does:

* Hydrogen is transported as a **passive scalar** on the pressure-model flow
  field; injection does not perturb the pressure solution.
* The flow is **isothermal**; a ±15 K seasonal swing moves every residence time
  by 5 to 10 %.
* Mixing at an injection point is **instantaneous**; the mixing length,
  of order 10² diameters, is not resolved at network scale.
* Strict observability requires a pressure measurement at **every** node, or the
  withdrawals at unmeasured nodes as known inputs; see `robustness_studies.py`.
* The framework is **verified**, against a converged plug-flow reference, an
  independent steady-state solver and a reference equation of state, but it is
  **not validated** against measured transmission data, because no public data
  set exists for a blended network.
