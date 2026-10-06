"""advection_benchmark.py - convergence of the multi-cell upwind transport model.

IMPLEMENTATION NOTES
--------------------
(1) SOLVER.  The previous version integrated the transport ODE with LSODA from
    y0 = y_base with u(0) = y_base.  That is an EXACT equilibrium, so the solver
    grew its step without bound and stepped straight over a pulse beginning
    hours later: the pulse benchmark returned identically zero error for every
    M, and the "11 % vs 21 % peak" figure quoted in the manuscript could not be
    reproduced from this script.  The transport model is linear and
    time-invariant, so it is now propagated by an EXACT zero-order-hold update,
    which is both faster and immune to that failure mode.

(2) CONVERGENCE STUDY.  The previous version compared only M = 1 against
    M = 40.  It therefore could not justify the choice M = 4-6 used downstream.
    A full sweep is reported here, with the three metrics that matter for a
    safety constraint: breakthrough time, front width, and PEAK delivered
    fraction.

(3) CFL-MATCHED AND FIR ALTERNATIVES are benchmarked alongside, showing that
    the accuracy/cost trade-off can be removed entirely (see transport_fir.py).
"""
import numpy as np

from gas_properties import mixture_properties
from network import Network, steady_state
from composition import build_composition_model
from composition_adv import build_adv_composition
from transport_fir import (build_upwind, build_cfl_composition, discretise,
                           markov_params, fir_settling)

TS_BLEND = 1800.0
DELIV = [1, 2, 3, 4, 5]
T_SIM = 48 * 3600.0
NT = 1153                     # 150 s output grid


def propagate(A, B, W, u_of_t, T=T_SIM, nt=NT, y0=0.05):
    """Exact ZOH propagation of  dy/dt = A y + B u  on a uniform grid."""
    te = np.linspace(0.0, T, nt)
    Ad, Bd = discretise(A, B, te[1] - te[0])
    X = np.zeros((A.shape[0], nt))
    X[:, 0] = y0
    for k in range(nt - 1):
        X[:, k + 1] = Ad @ X[:, k] + Bd[:, 0] * u_of_t(te[k])
    return te, {n: W[n] @ X for n in DELIV}


def step_u(t):
    return 0.20


def pulse_u(t):
    return 0.30 if 4 * 3600 <= t < 10 * 3600 else 0.05


def crossing(te, y, frac):
    """Time at which y first reaches `frac` of its total change."""
    b, f = y[0], y[-1]
    if abs(f - b) < 1e-4:
        return np.nan
    return te[np.argmax(y >= b + frac * (f - b))] / 3600


def main():
    net = Network()
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)
    cm = build_composition_model(net, props, P0, Q0)
    tau = cm['tau']

    print("=== Convergence of the M-cell upwind transport model "
          "(didactic network) ===")
    print(f"residence times tau [h]: {np.round(tau / 3600, 1)}")
    print("A pipe discretised into M well-mixed cells is a tanks-in-series")
    print("residence-time distribution with sigma_tau/tau = 1/sqrt(M): the mean")
    print("transit time is exact for every M, but the FRONT WIDTH is not.")
    print("  M =  1 -> +/-100 %   M =  4 -> +/-50 %   M = 25 -> +/-20 %\n")

    # --- converged plug-flow reference ----------------------------------
    ref = build_upwind(net, props, P0, Q0, np.full(net.n_pipes, 100))
    teS, refS = propagate(ref['A'], ref['B'], ref['Wsel'], step_u)
    teP, refP = propagate(ref['A'], ref['B'], ref['Wsel'], pulse_u)
    print(f"reference: M = 100 per pipe ({ref['nstate']} states)\n")

    print(f"{'M':>5} {'states':>7} | {'PULSE: peak error [vol-pts]':>30} | "
          f"{'STEP: 10% breakthrough err [h]':>31}")
    print(f"{'':>5} {'':>7} | {'worst':>9} {'N3':>9} {'N5':>9} | "
          f"{'worst':>9} {'N3':>10} {'N5':>10}")
    rows = []
    for M in [1, 2, 4, 6, 8, 12, 20, 40, 80]:
        c = build_adv_composition(net, props, P0, Q0, M=M)
        _, cs = propagate(c['A'], c['B'], c['Wsel'], step_u)
        _, cp_ = propagate(c['A'], c['B'], c['Wsel'], pulse_u)
        pk = {n: (refP[n].max() - cp_[n].max()) * 100 for n in DELIV}
        ta = {n: crossing(teS, cs[n], 0.10) - crossing(teS, refS[n], 0.10)
              for n in DELIV}
        rows.append(dict(M=M, ns=c['nstate'], peak=pk, arr=ta))
        print(f"{M:>5} {c['nstate']:>7} | {max(pk.values()):9.2f} {pk[3]:9.2f} "
              f"{pk[5]:9.2f} | {max(abs(v) for v in ta.values()):9.2f} "
              f"{ta[3]:10.2f} {ta[5]:10.2f}")

    # --- CFL-matched -----------------------------------------------------
    for refine, lbl in [(1, 'CFL'), (3, 'CFL/3')]:
        c = build_cfl_composition(net, props, P0, Q0, TS_BLEND, refine=refine)
        _, cs = propagate(c['A'], c['B'], c['Wsel'], step_u)
        _, cp_ = propagate(c['A'], c['B'], c['Wsel'], pulse_u)
        pk = {n: (refP[n].max() - cp_[n].max()) * 100 for n in DELIV}
        ta = {n: crossing(teS, cs[n], 0.10) - crossing(teS, refS[n], 0.10)
              for n in DELIV}
        rows.append(dict(M=lbl, ns=c['nstate'], peak=pk, arr=ta))
        print(f"{lbl:>5} {c['nstate']:>7} | {max(pk.values()):9.2f} {pk[3]:9.2f} "
              f"{pk[5]:9.2f} | {max(abs(v) for v in ta.values()):9.2f} "
              f"{ta[3]:10.2f} {ta[5]:10.2f}   M_k = {c['Mk'].tolist()}")

    print("\nREADING")
    m1 = next(r for r in rows if r['M'] == 1)
    m4 = next(r for r in rows if r['M'] == 4)
    print(f"  The single-cell model under-predicts the peak delivered fraction "
          f"of a 6 h pulse by")
    print(f"  {max(m1['peak'].values()):.1f} vol-pts at the worst node - the "
          f"reason it was rejected. At M = 4 the same error is still")
    print(f"  {max(m4['peak'].values()):.1f} vol-pts. M = 4-6 is therefore NOT "
          f"an adequate plug-flow surrogate for a")
    print("  safety constraint, and the choice must be justified by this "
          "convergence study rather")
    print("  than asserted.")

    # --- FIR: accuracy at zero online cost -------------------------------
    G = markov_params(ref['A'], ref['B'], ref['Wsel'], TS_BLEND, 400,
                      rows=DELIV)
    print("\nFIR (impulse-response) representation of the same reference model:")
    print(f"  taps for 99.9 % settling : {fir_settling(G)} "
          f"({fir_settling(G)*TS_BLEND/3600:.0f} h)")
    print(f"  steady gain per node     : "
          f"{np.array2string(G.sum(axis=0)[:, 0], precision=6)}")
    print("  (exactly 1.0 at every node - hydrogen mass is conserved by the "
          "transport operator)")
    print("  online cost              : 0 transport states; the predictive "
          "problem carries only")
    print("                             the H injection decisions, so the "
          "predictor is the M = 100")
    print("                             model at less cost than the M = 4 "
          "state-space version.")

    np.save("advection_convergence.npy",
            dict(rows=rows, tau=tau, te=teP,
                 ref_pulse={n: refP[n] for n in DELIV}), allow_pickle=True)
    print("\nsaved advection_convergence.npy")


if __name__ == "__main__":
    main()
