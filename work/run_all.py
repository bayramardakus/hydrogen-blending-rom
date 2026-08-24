"""run_all.py - reproduce every number and figure in the manuscript, in order.

Usage:  python run_all.py            (all steps)
        python run_all.py 3 4 7      (selected steps)

Each step is run in a fresh subprocess so that a failure is isolated and
reported rather than silently poisoning later steps.
"""
import subprocess, sys, time, os

STEPS = [
    ("extract_subnet.py",      "deterministic SciGRID_gas sub-network extraction"),
    ("i8_eos.py",              "GERG-2008 comparison of the mixing rules"),
    ("advection_benchmark.py", "transport discretisation convergence"),
    ("structural_analysis.py", "structure, estimator robustness, frozen flow, sensitivity"),
    ("realnet_pandapipes.py",  "code-to-code verification vs pandapipes"),
    ("realnet_reduction.py",   "reduction accuracy on the real network"),
    ("realnet_pressure.py",    "Case Study 3 pressure regulation"),
    ("rerun_blend_adv.py",     "didactic blend scheduling"),
    ("realnet_blend.py",       "Case Study 3 two-point blend scheduling"),
    ("energy_accounting.py",   "energy balance"),
    ("robustness_studies.py",  "sparse telemetry; green-H2 forecast error [not in the paper]"),
    ("reviewer_followups.py",  "two-way coupling magnitude; reversal threshold; uptake attribution"),
    ("fig_parity.py",          "pandapipes parity figure"),
    ("make_new_figures.py",    "figures revised in Revision 2"),
]


def main(sel):
    here = os.path.dirname(os.path.abspath(__file__))
    todo = STEPS if not sel else [STEPS[i - 1] for i in sel]
    fails = []
    t0 = time.time()
    for i, (script, what) in enumerate(todo, 1):
        print(f"\n{'='*78}\n[{i}/{len(todo)}] {script}  --  {what}\n{'='*78}",
              flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, script], cwd=here)
        dt = time.time() - t
        if r.returncode != 0:
            fails.append(script)
            print(f"  !! FAILED after {dt:.0f}s", flush=True)
        else:
            print(f"  -- ok ({dt:.0f}s)", flush=True)
    print(f"\n{'='*78}")
    print(f"{len(todo)-len(fails)}/{len(todo)} steps succeeded in "
          f"{(time.time()-t0)/60:.1f} min")
    if fails:
        print("FAILED:", ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]]))
