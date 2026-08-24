"""
verify.py
---------
Independent sanity checks + blend-ratio sweep producing the results table.
"""
import numpy as np
from numpy.linalg import eigvals
from gas_properties import mixture_properties
from network import Network, steady_state, build_rlc, build_reference
from experiment_validation import run_case

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)



def checks():
    net = Network(); props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)

    # 1) steady-state mass balance
    mb = (net.inc @ Q0 + net.demand_nom)[1:]
    print(f"[1] steady mass-balance max residual : {np.abs(mb).max():.2e} kg/s  (ok if ~0)")
    print(f"    total supply = {Q0[0]:.2f} kg/s , total demand = {net.demand_nom.sum():.2f} kg/s")

    # 2) RLC LTI stability: all eigenvalues must have negative real part
    A_c, B_c, C_out, info = build_rlc(net, props, P0, Q0)
    ev = eigvals(A_c)
    print(f"[2] RLC A-matrix max Re(eig) : {ev.real.max():.3e}  (stable if < 0)")

    # 3) reference conservation over a passive run (no disturbance): flows steady
    rhs, init_state, meta = build_reference(net, props, M=15, Q_basis=Q0)
    from scipy.integrate import solve_ivp
    y0 = init_state(P0, Q0)
    sol = solve_ivp(lambda t, y: rhs(t, y, net.p_source_nom, net.demand_nom),
                    [0, 1800], y0, method="LSODA", rtol=1e-6, atol=1e-3)
    idxN = meta['idx']['node']
    drift = np.abs(sol.y[idxN, -1] - sol.y[idxN, 0]).max() / 1e5
    print(f"[3] reference node-pressure drift at steady input : {drift:.2e} bar  (ok if small)")

    # 4) unit check: L,R,C give physical wave/time scales
    L, R, C = info['Lk'], info['Rk'], info['Cn']
    print(f"[4] RLC element ranges: L=[{L.min():.2e},{L.max():.2e}] "
          f"R=[{R.min():.2e},{R.max():.2e}] C=[{C[1:].min():.2e},{C[1:].max():.2e}]")


def blend_sweep():
    print("\nBlend-ratio sweep (pressure & flow accuracy, speed-up):")
    print(f"{'H2%':>5} {'a[m/s]':>7} {'HHV%':>6} {'p-MAPE%':>8} {'p-max[bar]':>10} "
          f"{'q-MAPE%':>8} {'speedup':>8}")
    rows = []
    for x in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        r = run_case(x, M_ref=20)
        sp = r['t_ref'] / r['t_rlc']
        print(f"{x*100:5.0f} {r['props']['a']:7.0f} {r['props']['hhv_ratio']*100:6.1f} "
              f"{r['mape']:8.3f} {r['maxerr']:10.3f} {r['fmape']:8.3f} {sp:8.0f}")
        # t_ref/t_rlc are stored, not just their ratio, so that Fig.~timing and
        # Table 4 are drawn from ONE timing run. Quoting a speed-up in the table
        # from one run and annotating the figure from another produced a 20 %
        # disagreement between the two in the previous version.
        rows.append(dict(h2=x*100, a=r['props']['a'], hhv=r['props']['hhv_ratio']*100,
                         pmape=r['mape'], pmax=r['maxerr'], qmape=r['fmape'],
                         nrmse=r['nrmse'], speedup=sp, rmse=r['rmse'],
                         t_ref=r['t_ref'], t_rlc=r['t_rlc'],
                         n_ref=r['n_ref'], n_rlc=r['n_rlc']))
    np.save("sweep.npy", rows, allow_pickle=True)
    return rows


if __name__ == "__main__":
    checks()
    blend_sweep()
