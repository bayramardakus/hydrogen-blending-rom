"""
experiment_validation.py
------------------------
Validate the reduced-order RLC model against the high-fidelity 1D transient
reference, for 5% and 25% hydrogen blends. Produces accuracy metrics, a
computation-time comparison, and the validation figures.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_properties import mixture_properties
from network import Network, steady_state, build_rlc, build_reference

# ------------------------------------------------------------------ scenario
T_END = 7200.0          # 2 hours
NODE_LEAK = 3           # leak/extra-demand node
LEAK_START, LEAK_RAMP = 1800.0, 300.0
LEAK_SIZE = 25.0        # kg/s additional withdrawal (leak / demand surge)
DEMAND_STEP_NODE = 5
DEMAND_STEP_START, DEMAND_STEP_RAMP = 4200.0, 300.0
DEMAND_STEP_SIZE = 15.0


def demand_profile(t, net):
    """Time-varying demand vector [kg/s]."""
    d = net.demand_nom.copy().astype(float)
    # leak / surge at NODE_LEAK
    r1 = np.clip((t - LEAK_START) / LEAK_RAMP, 0, 1)
    d[NODE_LEAK] += LEAK_SIZE * r1
    # later demand step at another node
    r2 = np.clip((t - DEMAND_STEP_START) / DEMAND_STEP_RAMP, 0, 1)
    d[DEMAND_STEP_NODE] += DEMAND_STEP_SIZE * r2
    return d


def run_case(x_h2, M_ref=20, make_plots=False, tag=""):
    net = Network()
    props = mixture_properties(x_h2)
    P0, Q0 = steady_state(net, props)
    u_src0 = net.p_source_nom
    d0 = net.demand_nom.copy()

    # ---- reduced RLC (LTI), simulated in deviation variables ----
    A_c, B_c, C_out, info = build_rlc(net, props, P0, Q0)
    nN, nP = net.n_nodes, net.n_pipes
    x0_abs = np.concatenate([P0[1:], Q0])          # non-source pressures + flows

    def rlc_rhs(t, dx):
        d = demand_profile(t, net)
        du = np.concatenate([[0.0], d - d0])       # p_source held at nominal
        return A_c @ dx + B_c @ du

    t_eval = np.linspace(0, T_END, 361)
    def _time(fn, n=3):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter()-t0)
        return np.median(ts), out
    t_rlc, sol_rlc = _time(lambda: solve_ivp(
        rlc_rhs, [0, T_END], np.zeros(info['ns']), t_eval=t_eval,
        method="RK45", rtol=1e-6, atol=1e-6))
    P_rlc = np.vstack([np.full_like(t_eval, P0[0]),           # source node
                       P0[1:, None] + sol_rlc.y[:nN-1, :]])   # nodes 1..5
    Q_rlc = Q0[:, None] + sol_rlc.y[nN-1:, :]

    # ---- high-fidelity reference (nonlinear, fine grid) ----
    rhs, init_state, meta = build_reference(net, props, M=M_ref, Q_basis=Q0)
    y0 = init_state(P0, Q0)

    def ref_rhs(t, y):
        d = demand_profile(t, net)
        return rhs(t, y, u_src0, d)

    t_ref, sol_ref = _time(lambda: solve_ivp(
        ref_rhs, [0, T_END], y0, t_eval=t_eval,
        method="LSODA", rtol=1e-6, atol=1e-3))
    idxN = meta['idx']['node']
    P_ref = sol_ref.y[idxN, :].copy()
    P_ref[net.source, :] = P0[0]
    # reference pipe flow = average of its faces
    Q_ref = np.zeros((nP, len(t_eval)))
    for k in range(nP):
        _, face_sl = meta['idx']['pipe'][k]
        Q_ref[k, :] = sol_ref.y[face_sl, :].mean(axis=0)

    # ---- error metrics on node pressures (non-source) ----
    Pr = P_ref[1:, :] / 1e5   # bar
    Pm = P_rlc[1:, :] / 1e5
    err = Pm - Pr
    rmse = np.sqrt(np.mean(err**2))                       # bar
    nrmse = rmse / (Pr.max() - Pr.min()) * 100            # % of range
    mape = np.mean(np.abs(err) / np.abs(Pr)) * 100        # %
    maxerr = np.max(np.abs(err))                          # bar
    # flow metrics
    ferr = Q_rlc - Q_ref
    frmse = np.sqrt(np.mean(ferr**2))
    # Denominator floored at 1 % of the largest pipe flow, NOT regularised by
    # adding a fixed 1 kg/s offset: the additive form flatters low-flow loop
    # chords, whose flow is small precisely where the reduced model is least
    # accurate. This matches realnet_reduction.py and what Section 5.3 states.
    qsc = np.maximum(np.abs(Q_ref), 0.01 * np.abs(Q_ref).max())
    fmape = np.mean(np.abs(ferr) / qsc) * 100

    res = dict(x_h2=x_h2, props=props, t=t_eval, P_ref=P_ref, P_rlc=P_rlc,
               Q_ref=Q_ref, Q_rlc=Q_rlc, rmse=rmse, nrmse=nrmse, mape=mape,
               maxerr=maxerr, frmse=frmse, fmape=fmape, t_ref=t_ref, t_rlc=t_rlc,
               n_ref=meta['n_state'], n_rlc=info['ns'], net=net, P0=P0, Q0=Q0)
    return res


if __name__ == "__main__":
    for x in [0.05, 0.25]:
        r = run_case(x, M_ref=20)
        print(f"\n=== H2 = {x*100:.0f} vol%  (a={r['props']['a']:.0f} m/s) ===")
        print(f"  states: reference={r['n_ref']}, RLC={r['n_rlc']}")
        print(f"  pressure  RMSE={r['rmse']:.4f} bar  NRMSE={r['nrmse']:.3f}%  "
              f"MAPE={r['mape']:.4f}%  max={r['maxerr']:.4f} bar")
        print(f"  flow      RMSE={r['frmse']:.3f} kg/s  MAPE={r['fmape']:.3f}%")
        print(f"  CPU: reference={r['t_ref']*1000:.1f} ms  RLC={r['t_rlc']*1000:.1f} ms "
              f" speedup={r['t_ref']/r['t_rlc']:.1f}x")
