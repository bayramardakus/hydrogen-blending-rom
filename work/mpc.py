"""
mpc.py
------
Three-layer control demonstration for the hydrogen-enriched pipeline network:

  Layer 1 (supervisory): constrained Model Predictive Control on the reduced
     RLC state-space, minimising  J = sum(x'Qx + u'Ru)  over a receding horizon,
     with a Kalman observer estimating the unmeasured (flow) states.
  Layer 2/3: the estimated control (compressor outlet pressure) is applied to
     the HIGH-FIDELITY 1D reference model, which plays the role of the physical
     pipeline (the plant).

We compare closed-loop MPC against an uncontrolled baseline (fixed compressor)
under a demand-surge / leak disturbance, and check the delivery-pressure limit.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_discrete_are
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gas_properties import mixture_properties
from network import Network, steady_state, build_rlc, build_reference

import os as _os
FIGDIR = _os.environ.get("PAPER_FIGS", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


def demand_profile(t, net):
    """More severe demand-surge / leak scenario for the control demonstration."""
    d = net.demand_nom.copy().astype(float)
    # large sustained leak/surge at node 3 (ramp 1800-2100 s)
    d[3] += 55.0 * np.clip((t - 1800.0)/300.0, 0, 1)
    # second surge at node 5 (ramp 4200-4500 s)
    d[5] += 30.0 * np.clip((t - 4200.0)/300.0, 0, 1)
    return d

rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                 "font.size": 11, "figure.dpi": 160, "savefig.dpi": 300,
                 "savefig.bbox": "tight"})
C_REF="#374151"; C_RLC="#2C7FB8"; C_ACC="#D95F0E"; C_OL="#B0357F"
FIG=FIGDIR

# ---- control / limits ----
TS = 120.0            # control sampling [s]
T_END = 7200.0
H = 15                # prediction horizon (steps) -> 30 min
P_MIN_BAR = 58.0      # minimum allowable delivery pressure [bar]
P_SRC_MIN, P_SRC_MAX = 60.0, 78.0   # compressor outlet bounds [bar]


def discretize(A, B, Ts):
    n, m = A.shape[0], B.shape[1]
    Mblk = np.zeros((n+m, n+m))
    Mblk[:n, :n] = A; Mblk[:n, n:] = B
    Md = expm(Mblk*Ts)
    return Md[:n, :n], Md[:n, n:]


def run_mpc(x_h2=0.25):
    net = Network(); props = mixture_properties(x_h2)
    P0, Q0 = steady_state(net, props)
    nN, nP = net.n_nodes, net.n_pipes
    A_c, B_c, C_out, info = build_rlc(net, props, P0, Q0)
    ns = info['ns']

    # split inputs: u_ctrl = p_source (col 0); d = demand (cols 1..nN)
    Bu_c = B_c[:, [0]]
    Bd_c = B_c[:, 1:]
    Ad, Bud = discretize(A_c, Bu_c, TS)
    _, Bdd = discretize(A_c, Bd_c, TS)
    Cd = C_out                      # measure non-source node pressures (deviation)

    # ---- Kalman observer (steady-state) ----
    Qk = np.eye(ns)*1e-2; Rk = np.eye(nN-1)*1e-3
    Pobs = solve_discrete_are(Ad.T, Cd.T, Qk, Rk)
    Lobs = Pobs @ Cd.T @ np.linalg.inv(Cd @ Pobs @ Cd.T + Rk)

    # ---- MPC weights ----
    Qx = np.zeros((ns, ns))
    for i in range(nN-1):            # penalise node-pressure deviations (in bar^2)
        Qx[i, i] = 1.0
    Ru = np.array([[0.05]])          # penalise compressor effort
    # terminal weight from DARE
    Qf = solve_discrete_are(Ad, Bud, Qx, Ru)

    # scale: work in bar for pressures to keep QP well conditioned
    SC = 1e5
    AdS, BudS, BddS = Ad, Bud*SC, Bdd*SC   # states in Pa; convert later

    def build_qp():
        x0 = cp.Parameter(ns)
        dseq = cp.Parameter((H, nN))       # demand deviation forecast [kg/s]
        u = cp.Variable((H, 1))
        x = cp.Variable((H+1, ns))
        s = cp.Variable((H, nN-1), nonneg=True)   # soft slack on Pmin
        cons = [x[0] == x0]
        cost = 0
        for k in range(H):
            cons += [x[k+1] == Ad @ x[k] + Bud @ u[k] + Bdd @ dseq[k]]
            # compressor bound (absolute): P0_src + u in [min,max]
            cons += [u[k] >= (P_SRC_MIN*1e5 - P0[0]),
                     u[k] <= (P_SRC_MAX*1e5 - P0[0])]
            # delivery pressure soft lower bound at every non-source node
            for i in range(nN-1):
                cons += [P0[i+1] + x[k+1, i] >= P_MIN_BAR*1e5 - s[k, i]]
            cost += cp.quad_form(x[k+1, :nN-1]/SC, np.eye(nN-1)) \
                    + Ru[0,0]*cp.square(u[k]/SC) + 1e4*cp.sum(s[k]/SC)
        prob = cp.Problem(cp.Minimize(cost), cons)
        return prob, x0, dseq, u

    prob, p_x0, p_d, v_u = build_qp()

    # ---- plant = high-fidelity reference ----
    rhs, init_state, meta = build_reference(net, props, M=15, Q_basis=Q0)
    idxN = meta['idx']['node']

    def simulate(controlled):
        y0 = init_state(P0, Q0)
        xhat = np.zeros(ns)
        nsteps = int(T_END/TS)
        tlog, Psrc_log, Pmin_log, Pnodes_log = [], [], [], []
        yc = y0.copy()
        u_abs = P0[0]
        for step in range(nsteps):
            t = step*TS
            # measurement (deviation of node pressures)
            Pmeas = yc[idxN].copy(); Pmeas[net.source] = P0[0]
            ymeas = (Pmeas[1:] - P0[1:])
            # observer correction
            xhat = xhat + Lobs @ (ymeas - Cd @ xhat)
            if controlled:
                # demand forecast over horizon (assume measured, persistence)
                dseq = np.zeros((H, nN))
                for j in range(H):
                    dseq[j] = demand_profile(t+j*TS, net) - net.demand_nom
                p_x0.value = xhat; p_d.value = dseq
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                           max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
                du = float(v_u.value[0, 0])
                u_abs = P0[0] + du
            else:
                u_abs = P0[0]
            # apply to plant over one Ts
            d_now = demand_profile(t+TS/2, net)
            sol = solve_ivp(lambda tt, yy: rhs(tt, yy, u_abs, d_now),
                            [t, t+TS], yc, method="LSODA", rtol=1e-6, atol=1e-3)
            yc = sol.y[:, -1]
            # observer time update
            dnow_dev = d_now - net.demand_nom
            xhat = Ad @ xhat + Bud.flatten()*(u_abs-P0[0]) + Bdd @ dnow_dev
            # log
            Pn = yc[idxN].copy(); Pn[net.source] = u_abs
            tlog.append((t+TS)/3600.0)
            Psrc_log.append(u_abs/1e5)
            Pmin_log.append(Pn[1:].min()/1e5)
            Pnodes_log.append(Pn[1:]/1e5)
        return (np.array(tlog), np.array(Psrc_log), np.array(Pmin_log),
                np.array(Pnodes_log))

    ol = simulate(False)
    cl = simulate(True)
    return net, P0, ol, cl


def fig_mpc(net, P0, ol, cl):
    t_ol, ps_ol, pmin_ol, _ = ol
    t_cl, ps_cl, pmin_cl, _ = cl
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axs[0]
    ax.plot(t_ol, pmin_ol, color=C_OL, lw=2, label="Uncontrolled (fixed compressor)")
    ax.plot(t_cl, pmin_cl, color=C_RLC, lw=2, label="MPC on reduced model")
    ax.axhline(P_MIN_BAR, color=C_ACC, lw=1.4, ls="--", label=f"Min. delivery limit ({P_MIN_BAR:.0f} bar)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Minimum node pressure [bar]")
    ax.set_title("(a) Delivery-pressure regulation"); ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax = axs[1]
    ax.plot(t_cl, ps_cl, color=C_RLC, lw=2, label="MPC compressor set-point")
    ax.plot(t_ol, ps_ol, color=C_OL, lw=1.6, ls="--", label="Uncontrolled")
    ax.axhline(P_SRC_MAX, color="0.6", lw=1, ls=":"); ax.axhline(P_SRC_MIN, color="0.6", lw=1, ls=":")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Compressor outlet pressure [bar]")
    ax.set_title("(b) Control action"); ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_mpc.png"); plt.close(fig)
    print("saved fig_mpc.png")


if __name__ == "__main__":
    net, P0, ol, cl = run_mpc(0.25)
    print(f"Uncontrolled min pressure: {ol[2].min():.2f} bar")
    print(f"MPC min pressure:          {cl[2].min():.2f} bar  (limit {P_MIN_BAR})")
    print(f"MPC max compressor:        {cl[1].max():.2f} bar")
    fig_mpc(net, P0, ol, cl)
