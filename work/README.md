# Composition-Aware Reduced-Order Modelling of Real-Time Hydrogen Blending
## Reproduction package (Revision 1)

Every figure, table and number in the manuscript is produced by the scripts in
this directory. Nothing is hand-entered.

---

## 1. Requirements

```
python >= 3.10
numpy scipy matplotlib pandas networkx cvxpy CoolProp pandapipes
```

```bash
pip install numpy scipy matplotlib pandas networkx cvxpy CoolProp pandapipes
```

CVXPY must have the CLARABEL solver available (it ships with CVXPY ≥ 1.4).

## 2. Data

The SciGRID_gas IGGIELGN data set (v1.1.2) is **not** redistributed here. Download
it from Zenodo and unpack it so that the CSV files sit in `../scigrid/data/`:

```
https://doi.org/10.5281/zenodo.4767098
```

```
<repo>/
  scigrid/data/IGGIELGN_PipeSegments.csv
  scigrid/data/IGGIELGN_Nodes.csv
  ...
  work/            <- this directory
  ms/figs/         <- figure output
```

## 3. Reproducing everything

```bash
python run_all.py            # ~45 min on one core
```

or step by step, in this order (later steps consume earlier `.npy` outputs):

| # | Script | Produces | Manuscript |
|---|--------|----------|-----------|
| 1 | `extract_subnet.py` | `scigrid_subnet.json` | §5.1 |
| 2 | `i8_eos.py` | GERG-2008 comparison | §6.1, Table 3 |
| 3 | `advection_benchmark.py` | `advection_convergence.npy` | §6.2, Fig. 8a,b |
| 4 | `structural_analysis.py` | `structural_analysis.npy` | §6.2, §6.6, Figs. 8c, 12 |
| 5 | `realnet_pandapipes.py` | `realnet_pandapipes.npy` | §6.3, Fig. 9 |
| 6 | `realnet_reduction.py` | `realnet_reduction.npy` | §6.4 |
| 7 | `realnet_pressure.py` | `realnet_pressure_results.npy` | §6.9, Fig. 14 |
| 8 | `rerun_blend_adv.py` | `blend_adv_results.npy` | §6.5, Fig. 11 |
| 9 | `realnet_blend.py` | `realnet_blend_results.npy` | §6.9, Fig. 13 |
| 10 | `energy_accounting.py` | `energy_accounting.npy` | §6.8 |
| 11 | `verify.py` | `sweep.npy` | §6.4, Table 4, Fig. 10a |
| 12 | `parameter_studies.py` | `backoff_study.npy`, `wobbe_variant.npy`, `surge_sweep.npy` | §4.3, §6.9, §6.10, Fig. 15 |
| 13 | `robustness_studies.py` | `robustness_studies.npy` | not in the paper; kept for the review record |
| 14 | `reviewer_followups.py` | `reviewer_followups.npy` | §6.2, §6.8, §7 |
| 15 | `fig_parity.py` | pandapipes parity figure | Fig. 9 |
| 16 | `make_new_figures.py` | all revised figures | Figs. 7, 10–15 |

Set `PAPER_FIGS` to redirect figure output; the default is `../ms/figs`.

## 4. Determinism

`extract_subnet.py` returns a **bit-identical** `scigrid_subnet.json` on repeated
runs against the archived data set:

```bash
for i in 1 2 3; do python extract_subnet.py > /dev/null; md5sum scigrid_subnet.json; done
```

This required two fixes over the previous version, because graph traversal in
NetworkX is not order-stable:

* node indices are assigned by **sorting on the SciGRID_gas identifier**;
* each pipe is written with a **canonical endpoint orientation** (`a < b`), which
  fixes the sign convention of the incidence matrix and therefore the reported
  sign of every pipe flow.

Without them a re-run reproduced the same topology but relabelled 18 of 22 nodes.
For the same reason **every disturbance scenario is anchored by a rule on a
physical property**, never by a node number:

* `realnet.major_offtakes(net, k)` ,  the k largest withdrawals;
* `realnet.weakest_nodes(net, P0, k)` ,  the k nodes with the smallest nominal
  margin to the delivery floor;
* `realnet.far_nodes(net, k)` ,  the k nodes furthest from the supply.

## 5. What changed relative to the original submission

Each item below is a defect found and corrected in our own package; the affected
manuscript section is given in brackets.

| Area | Defect | Fix |
|---|---|---|
| `advection_benchmark.py` | Integrated from an exact equilibrium, so LSODA grew its step without bound and stepped over the pulse; the pulse benchmark returned identically zero. The "11 % vs 21 %" figure was not reproducible. | Exact zero-order-hold propagation of the LTI transport model. [§6.2] |
| transport discretisation | `M = 4–6` was asserted, not derived; it retains 75 % of the error for which `M = 1` was rejected. | Full convergence study; predictor moved to the impulse-response form (`transport_fir.py`), which is the same LTI system in a different realisation and carries no online states. [§2.3, §6.2] |
| `realnet_blend.py` | The "plant" was advanced with the controller's own discrete model, making zero violation a tautology. | Independent plant: finer transport model on a **time-varying** flow field with τ and mixing weights recomputed, reversal handling, actuator noise. [§6.9] |
| blend constraint | Imposed at exactly 20 vol% with no margin, against a predictor whose own error had been measured. | Explicit back-off, Eq. (12), sized from the measured error envelope. [§4.3, Fig. 15] |
| `i9_i5.py` (frozen flow) | Reported max-over-nodes of the **mean over time** as a worst case and as a bound. | True worst case over time and nodes; scenario set extended to genuine reversal. [§6.2] |
| `network.py` (steady solve) | Residual mixed O(1) kg/s with O(10¹¹) Pa² blocks, so the solve failed even from the exact solution and could not be continued into reversal. | Non-dimensionalised residual + `steady_continuation`. |
| `i8_eos.py` | Compared an M = 18 g/mol model against a **pure-methane** GERG mixture, and formed the reference wave speed from the model's own R_s. | Consistent base gas; a_GERG = √(p/ρ_GERG). Density deviation 14.7 % → **2.3 %**. [§6.1] |
| observer tuning | `R = 1e-3·I` with states in Pa ⇒ σ_v = 0.03 Pa, ‖L‖∞ = 1; flow estimates were amplified noise. | Scaled model; R = σ_v²I from a stated transducer class, Q = B_dΣ_dB_dᵀ from demand-forecast error. [§4.1, §6.6] |
| Gramian tests | Run on the raw model (cond ≈ 10¹⁶); printed rank deficiency for properties the manuscript claimed. | Non-dimensionalised; observability by rank with margin, controllability stated precisely (output-controllability 8/21 with one actuator; unit DC gain to every node). [§6.6] |
| terminal cost | Claimed as a Riccati terminal cost; unused in one script, a rescaled sub-block in another, absent in a third. | Claim withdrawn; replaced by the argument that applies ,  A is Hurwitz, so receding-horizon control of an open-loop stable plant with bounded inputs is stable. [§4.2] |
| horizons | 18 h horizon and 18 h post-injection window against a network whose slowest branch has τ = 81.2 h. | Control horizon 24 h, constraint horizon 162 h (99.9 % settling), run length 244 h. [§4.3, §5.1] |
| flow error metric | `|Δq| / (|q| + 1.0)` flattered low-flow loop chords. | Denominator floored at 1 % of max flow. [§5.3] |
| `realnet_reduction.py` | A failed stiff reference solve returned fewer time points and an artificially small error and speed-up. | Hard integrity check on `sol.success` and point count. |
| `energy_accounting.py` | Isothermal work described as a "conservative upper bound". | It is the **lower** bound; polytropic shaft work at η = 0.80 reported. [§6.8] |
| repository | Ten scripts had hard-coded absolute output paths and would not run elsewhere. | Repository-relative, overridable by `PAPER_FIGS`. |
| references | 14 of 41 entries had no authors; one DOI pointed to an unrelated paper. | All 41 verified against publisher records. |

## 5b. Second consistency pass (manuscript against code)

A full cross-check of every number in the manuscript against the result caches
found a further set of defects, all of the same family: **caches left over from
an intermediate code state, quoted in the text as if current**. Each is listed
with what was wrong and what now guarantees it stays right.

| Area | Defect | Fix |
|---|---|---|
| Table 4 (didactic accuracy) | The whole table came from a run predating the corrected reference solve. Pressure MAPE was reported up to 0.412 % where the code gives 0.049 %, and the speed-up as ~200x where it is ~500x. The manuscript's own Fig. 10 disagreed with its own Table 4. | Table regenerated from `verify.py`. `sweep.npy` now also stores `t_ref`/`t_rlc`, and `make_figures.fig_timing` reads the timings **from that same file**, so figure and table can no longer diverge. |
| `experiment_validation.py` | Flow error normalised as `|dq|/(|q|+1)`, the additive regularisation §5.3 explicitly disavows. | Denominator floored at 1 % of the largest pipe flow, as `realnet_reduction.py` already did and as the text states. |
| `backoff_study.npy` | Generated against neither the nominal plant nor the present independent plant, so Fig. 15 disagreed with the headline uptake of §6.9 by 6 points. | Regenerated by `parameter_studies.py` against the independent plant. The Delta = 2 vol-pt row now reproduces the §6.9 result (36.3 % uptake, 18.57 vol% peak, 1.43 pt margin) exactly. |
| `wobbe_variant.npy` | Same defect: the 20 vol% row read 42.1 % against the 36.3 % reported two sections earlier. | Regenerated on the same independent plant; the 20 vol% row now reads 36.6 %. |
| `backoff_study.npy`, `wobbe_variant.npy`, `surge_sweep.npy` | Three result files with **no script** that produced them, in a paper whose data statement promises the opposite. | `parameter_studies.py` added; it regenerates all three. |
| §6.5 controller comparison | Quoted 57.3 bar / 20 % / 58.2 / 59.2 bar against cached values of 56.67 / 35.0 / 58.00 / 58.55, and claimed the PID "holds the limit" when it violates it 1.7 % of the time. | Corrected from `mpc_ana.npy`; the PID's residual violation is now stated. |
| §6.6 scalability | "assembly <=0.13 s, solve <=0.17 s" against 0.024 s and 0.251 s; "three orders of magnitude" inside the interval when it is 436x. | Corrected; the claim is now "more than two orders of magnitude", stated in the abstract and conclusions too. |
| §6.8 energy balance | 15 GWh against 9.2 GWh from a fresh run; 12.3 MW and 0.016 % against 12.8 MW and 0.021 %. | Re-run and corrected. |
| Fig. 12 caption | Described a panel (c) that the plotting code had already deleted. | Caption corrected; the observer is characterised in Fig. 18. |
| several captions | Described encodings the code does not use (star/diamond markers, axis ranges, "sign and magnitude" for a panel plotting magnitudes, "bang-bang" for a ramp-limited QP). | Each caption checked against its generating function and corrected. |
| Eq. (4) | Carried a terminal term `x_H' Qf x_H` that is defined nowhere, valued nowhere, and disclaimed sixteen lines later. | Removed; the slack penalty that the code actually applies is written instead. |
| §3, §4.1 | `Q`,`R` denoted both the MPC weights and the Kalman covariances. | Filter covariances renamed `Q_w`, `R_v` throughout, and added to the nomenclature. |
| figures | Text clipped off the canvas edge (Figs. 7, 18), annotations buried under data (Figs. 7, 13), a threshold annotation colliding with a title (Fig. 11), type below the 7 pt floor, and a 0.4 in dead band under four figures that bypassed `figstyle.save`. | Each regenerated; all figures now go through `figstyle.save`. |

## 6. File map

**Model**
- `gas_properties.py` ,  NG/H₂ mixture properties, Colebrook–White friction
- `network.py` ,  didactic network; scaled steady solve; RLC reduction; 1-D reference
- `realnet.py` ,  real SciGRID_gas sub-network; rule-based scenario node selection
- `extract_subnet.py` ,  deterministic sub-network extraction from IGGIELGN
- `composition.py`, `composition_adv.py` ,  hydrogen transport (1-cell, M-cell)
- `transport_fir.py` ,  per-pipe and CFL-matched upwind; impulse-response form
- `estimator.py` ,  state scaling; physically-tuned steady-state Kalman filter

**Studies**
- `i8_eos.py` ,  GERG-2008 comparison
- `advection_benchmark.py` ,  transport discretisation convergence
- `structural_analysis.py` ,  controllability/observability, estimator robustness, frozen-flow envelope, roughness and temperature sensitivity
- `robustness_studies.py` ,  sparse pressure telemetry; green-hydrogen forecast error.
  **Not reported in the manuscript**: no reviewer asked for either study, and a
  revision is not the place to widen scope. Kept here for the review record.
- `reviewer_followups.py` ,  magnitude of the neglected two-way coupling; demand
  amplitude at which a loop chord reverses on the real network; attribution of
  the change in reported green-hydrogen uptake between revisions
- `parameter_studies.py` ,  back-off sweep, Wobbe-band variant, contingency-magnitude sweep
- `sensitivity_analysis.py` ,  standalone sensitivity package (superset of the above, kept for the review record)
- `realnet_pandapipes.py`, `pandapipes_validation.py` ,  code-to-code verification
- `realnet_reduction.py`, `accuracy_at_scale.py`, `verify.py` ,  reduction accuracy
- `scalability.py` ,  cost versus network size

**Control**
- `realnet_pressure.py` ,  Case Study 3 pressure regulation
- `realnet_blend.py` ,  Case Study 3 two-point blend scheduling
- `rerun_blend_adv.py` ,  didactic blend scheduling
- `mpc.py`, `mpc_analysis.py`, `blend_mpc.py` ,  didactic controller studies
- `energy_accounting.py` ,  energy balance

**Figures**
- `make_new_figures.py` ,  figures revised for this revision
- `make_figures.py`, `fig_*.py` ,  figures unchanged from the original submission
- `figstyle.py` ,  shared style

## 7. Known scope limits

Stated in §7 and Table 6 of the manuscript, and repeated here so the code is not
read as claiming more than it does:

* Hydrogen is transported as a **passive scalar** on the pressure-model flow
  field; injection does not perturb the pressure solution.
* The flow is **isothermal**; a ±15 K seasonal swing moves every residence time
  by 5–10 %.
* Mixing at an injection point is **instantaneous**; the mixing length
  (O(10²) diameters) is not resolved at network scale.
* Strict observability requires a pressure measurement at **every** node, or the
  withdrawals at unmeasured nodes as known inputs; see `robustness_studies.py`.
* The framework is **verified** ,  against a converged plug-flow reference, an
  independent steady-state solver and a reference equation of state ,  but not
  **validated** against measured transmission data, because no public data set
  exists for a blended network.
