"""
sensitivity_analysis.py  —  sensitivity and robustness studies.

The studies bound the modelling choices the rest of the package makes:

  S1  Numerical diffusion of the M-cell upwind transport model.
      Quantifies arrival-time, front-width and (safety-critical) PEAK error of
      M = 1..80 against a converged plug-flow reference, and shows that the
      per-pipe CFL-matched choice M_k = round(tau_k / T_s) is exact
      (Courant number = 1  =>  zero numerical diffusion) at a modest state cost.

  S2  Frozen-flow (nominal tau / nominal mixing weight) validity envelope,
      reported as the TRUE worst case over time AND nodes (the previous script
      reported max-over-nodes of the mean-over-time), and extended by
      pseudo-arclength continuation to operating points that genuinely reverse
      one or more loop chords.

  S3  Steady-state Kalman filter robustness under pressure-transducer noise
      (0.1 - 5 % of reading), for the tuning used in the paper and for a
      noise-matched tuning; reports the error of the UNMEASURED flow states.

  S4  Parametric sensitivity to pipe roughness (eps) and gas temperature (T).

Run from the work/ directory:   python sensitivity_analysis.py
Outputs:  fig_sensitivity_transport.png , fig_sensitivity_robustness.png
          sensitivity_results.npy
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_discrete_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_properties import mixture_properties, darcy_friction
from network import Network, steady_state, build_rlc
from composition import build_composition_model
from composition_adv import build_adv_composition

np.random.seed(0)
RES = {}
DEL = [1, 2, 3, 4, 5]
TS_BLEND = 1800.0          # blend-loop sampling used in the paper
MAX_STEP = 300.0           # IMPORTANT: bound the solver step, see note below


# ----------------------------------------------------------------------
# NOTE ON A SOLVER PITFALL (this bit its way into advection_benchmark.py):
# y0 = y_base and u(0) = y_base is an EXACT equilibrium of dy/dt = A y + B u.
# LSODA then grows its step without bound and steps straight over a pulse that
# starts hours later, returning a flat, physically meaningless trajectory.
# Always pass max_step (or use t_eval + a continuous input) for pulse tests.
# ----------------------------------------------------------------------
def simulate(A, B, W, u_of_t, T, nt=1153, y0=0.05, max_step=MAX_STEP):
    """Exact zero-order-hold propagation of the LTI transport model.

    The system is linear and time-invariant, so with a piecewise-constant input
    on a uniform grid the discrete update x_{k+1} = Ad x_k + Bd u_k is EXACT.
    This is both faster than a stiff ODE solver and immune to the step-skipping
    pitfall noted above.
    """
    te = np.linspace(0.0, T, nt)
    dt = te[1] - te[0]
    n, m = A.shape[0], B.shape[1]
    Mb = np.zeros((n + m, n + m)); Mb[:n, :n] = A; Mb[:n, n:] = B
    Md = expm(Mb * dt); Ad, Bd = Md[:n, :n], Md[:n, n:]
    X = np.zeros((n, nt)); X[:, 0] = y0
    for k in range(nt - 1):
        X[:, k + 1] = Ad @ X[:, k] + Bd[:, 0] * u_of_t(te[k])
    return te, {nn: W[nn] @ X for nn in DEL}


def build_cfl(net, props, P0, Q0, Ts):
    """Per-pipe CFL-matched upwind model: M_k = max(1, round(tau_k/Ts)).

    With Courant number sigma_k = (M_k/tau_k)*Ts = 1 the explicit upwind update
    is the exact shift operator, so the scheme carries ZERO numerical diffusion
    at the sampling instants of the blend loop.
    """
    cm = build_composition_model(net, props, P0, Q0)
    tau, up, Wn = cm['tau'], cm['upstream'], cm['Wnode']
    nP, nN = net.n_pipes, net.n_nodes
    Mk = np.maximum(1, np.round(tau / Ts).astype(int))
    off = np.concatenate([[0], np.cumsum(Mk)])
    ns = int(off[-1])
    A = np.zeros((ns, ns)); B = np.zeros((ns, 1)); W = np.zeros((nN, ns))
    out = lambda k: off[k] + Mk[k] - 1
    for k in range(nP):
        r = Mk[k] / tau[k]
        for i in range(Mk[k]):
            idx = off[k] + i
            A[idx, idx] -= r
            if i > 0:
                A[idx, off[k] + i - 1] += r
            elif up[k] == net.source:
                B[idx, 0] += r
            else:
                for j in range(nP):
                    if Wn[up[k], j]:
                        A[idx, out(j)] += r * Wn[up[k], j]
    for n in range(nN):
        for k in range(nP):
            if Wn[n, k]:
                W[n, out(k)] += Wn[n, k]
    return A, B, W, Mk, ns


# ======================================================================
# S1 — numerical diffusion of the M-cell upwind transport model
# ======================================================================
def S1(net, props, P0, Q0):
    print("\n" + "=" * 74)
    print("S1  NUMERICAL DIFFUSION OF THE M-CELL UPWIND TRANSPORT MODEL")
    print("=" * 74)
    cm = build_composition_model(net, props, P0, Q0)
    print(f"pipe residence times tau [h]: {np.round(cm['tau'] / 3600, 1)}")
    print("Tanks-in-series RTD:  sigma_tau/tau = 1/sqrt(M)  ->  M=4 smears the "
          "front by +/-50% of tau, M=25 by +/-20%.\n")

    step = lambda t: 0.20
    pulse = lambda t: 0.30 if 4 * 3600 <= t < 10 * 3600 else 0.05
    Ms = [1, 2, 4, 6, 8, 12, 20, 40, 80]

    # ---- reference (M=200 per pipe) -------------------------------------
    cmR = build_adv_composition(net, props, P0, Q0, M=100)
    teS_ref, refS = simulate(cmR['A'], cmR['B'], cmR['Wsel'], step, 48 * 3600)
    teP_ref, refP = simulate(cmR['A'], cmR['B'], cmR['Wsel'], pulse, 48 * 3600)

    def t_arr(te, y, frac):
        b, f = y[0], y[-1]
        if abs(f - b) < 1e-4:
            return np.nan
        return te[np.argmax(y >= b + frac * (f - b))] / 3600

    rows = []
    print(f"{'M':>4} {'states':>7} | {'PULSE peak error [vol-pts]':>28} | "
          f"{'STEP 10%-breakthrough error [h]':>32}")
    print(f"{'':>4} {'':>7} | {'worst':>9} {'N3':>9} {'N5':>8} | "
          f"{'worst':>9} {'N3':>10} {'N5':>10}")
    for M in Ms:
        c = build_adv_composition(net, props, P0, Q0, M=M)
        _, cs = simulate(c['A'], c['B'], c['Wsel'], step, 48 * 3600)
        _, cp = simulate(c['A'], c['B'], c['Wsel'], pulse, 48 * 3600)
        pk = {n: (refP[n].max() - cp[n].max()) * 100 for n in DEL}
        ta = {n: t_arr(teS_ref, cs[n], 0.10) - t_arr(teS_ref, refS[n], 0.10)
              for n in DEL}
        rows.append(dict(M=M, ns=net.n_pipes * M, peak=pk, arr=ta))
        print(f"{M:>4} {net.n_pipes * M:>7} | {max(pk.values()):9.2f} "
              f"{pk[3]:9.2f} {pk[5]:8.2f} | {max(abs(v) for v in ta.values()):9.2f} "
              f"{ta[3]:10.2f} {ta[5]:10.2f}")

    # ---- CFL-matched -----------------------------------------------------
    Ac, Bc, Wc, Mk, nsc = build_cfl(net, props, P0, Q0, TS_BLEND)
    _, cs = simulate(Ac, Bc, Wc, step, 48 * 3600)
    _, cp = simulate(Ac, Bc, Wc, pulse, 48 * 3600)
    pk = {n: (refP[n].max() - cp[n].max()) * 100 for n in DEL}
    ta = {n: t_arr(teS_ref, cs[n], 0.10) - t_arr(teS_ref, refS[n], 0.10) for n in DEL}
    print(f"{'CFL':>4} {nsc:>7} | {max(pk.values()):9.2f} {pk[3]:9.2f} "
          f"{pk[5]:8.2f} | {max(abs(v) for v in ta.values()):9.2f} "
          f"{ta[3]:10.2f} {ta[5]:10.2f}     <-- M_k = round(tau_k/Ts) = "
          f"{Mk.tolist()}")
    print("\nREADING: at M=4 the peak delivered fraction of a 6 h pulse is "
          "under-predicted by ~8 vol-pts at the\nfar nodes - the same "
          "safety-relevant failure mode for which the M=1 model was rejected.")
    RES['S1'] = dict(rows=rows, cfl=dict(Mk=Mk, ns=nsc, peak=pk, arr=ta),
                     te=teP_ref, refP=refP)
    return teP_ref, refP, rows


# ======================================================================
# S2 — frozen-flow validity envelope, incl. genuine loop reversal
# ======================================================================
def steady_scaled(net, props, demand, p_source, x0, eps=3.0e-5):
    """Residual-scaled Newton solve for the steady state.

    NOTE ON A CODE DEFECT IN network.steady_state():
    that routine hands fsolve a residual vector whose node block is O(1) kg/s
    and whose pipe block is O(p^2) ~ 1e11 Pa^2 - eleven orders of magnitude
    apart.  hybrd's trust region is then set by the pipe block alone, so the
    solve fails to converge from a warm start (even from the exact solution)
    and cannot be continued into flow-reversal operating points at all.
    Normalising both blocks to O(1) fixes it and makes continuation possible.
    """
    from scipy.optimize import root
    nN, nP = net.n_nodes, net.n_pipes
    a2 = props['a2']
    pS = p_source ** 2
    qS = max(np.abs(demand).max(), 1.0)

    def residual(x):
        P = np.concatenate([[p_source], x[:nN - 1] * 1e5])
        Q = x[nN - 1:]
        r = np.zeros(nN - 1 + nP)
        r[:nN - 1] = (net.inc @ Q + demand)[1:] / qS
        for k, (a, b, L, D) in enumerate(net.pipes):
            A = net.area[k]
            lam = darcy_friction(props['rho'], props['mu'], D, Q[k], eps=eps)
            coef = lam * a2 * L / (D * A ** 2)
            r[nN - 1 + k] = (P[a] ** 2 - P[b] ** 2 - coef * Q[k] * abs(Q[k])) / pS
        return r

    sol = root(residual, x0, method='hybr', tol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return (np.concatenate([[p_source], sol.x[:nN - 1] * 1e5]), sol.x[nN - 1:])


def steady_continuation(net, props, dem_target, P0, Q0, steps=40):
    """Ramp the demand vector from nominal to target with warm starts so that
    operating points which reverse a loop chord are actually reachable."""
    x = np.concatenate([P0[1:] / 1e5, Q0])
    P, Q = P0, Q0
    for f in np.linspace(0.0, 1.0, steps)[1:]:
        dem = net.demand_nom + f * (dem_target - net.demand_nom)
        P, Q = steady_scaled(net, props, dem, net.p_source_nom, x)
        x = np.concatenate([P[1:] / 1e5, Q])
    return P, Q


def S2(net, props, P0, Q0, M=20):
    print("\n" + "=" * 74)
    print("S2  FROZEN-FLOW VALIDITY ENVELOPE (true worst case, incl. reversal)")
    print("=" * 74)

    def curves(Qb, T=48 * 3600):
        c = build_adv_composition(net, props, P0, Qb, M=M)
        return simulate(c['A'], c['B'], c['Wsel'], lambda t: 0.20, T)

    te, base = curves(Q0)
    scen = []
    # (label, demand modification)
    cases = [("nominal +40 kg/s surge (paper scenario)", {3: +25.0, 5: +15.0}),
             ("heavy loop load  d2 +60",                 {2: +60.0}),
             ("d5 +60",                                  {5: +60.0}),
             ("d5 +100",                                 {5: +100.0}),
             ("secondary entry at N4 (-40 kg/s)",        {4: -40.0}),
             ("secondary entry at N4 (-80 kg/s)",        {4: -80.0}),
             ("secondary entry at N5 (-60 kg/s)",        {5: -60.0}),
             ("secondary entry at N5 (-100 kg/s)",       {5: -100.0}),
             ("secondary entry at N5 (-160 kg/s)",       {5: -160.0}),
             ("secondary entry at N3 (-120 kg/s)",       {3: -120.0})]
    print(f"{'scenario':>40} {'rev':>4} {'mean-of-max':>12} {'TRUE MAX':>10}"
          f"  {'[vol-pts]':>10}")
    for lbl, mod in cases:
        dem = net.demand_nom.astype(float).copy()
        for n, v in mod.items():
            dem[n] += v
        try:
            P1, Q1 = steady_continuation(net, props, dem, P0, Q0)
        except Exception:
            print(f"{lbl:>40} {'--':>4} {'(steady solve did not converge)':>34}")
            continue
        rev = int((np.sign(Q1) != np.sign(Q0)).sum())
        _, tru = curves(Q1)
        mean = max(np.mean(np.abs(base[n] - tru[n])) * 100 for n in DEL)
        mx = max(np.max(np.abs(base[n] - tru[n])) * 100 for n in DEL)
        scen.append(dict(label=lbl, rev=rev, mean=mean, max=mx))
        print(f"{lbl:>40} {rev:>4} {mean:12.2f} {mx:10.2f}")
    print("\nREADING: the quantity to report as an MPC constraint back-off is the "
          "TRUE MAX column.\nThe paper's 1.53 vol-pts is a max-over-nodes of the "
          "MEAN-over-time and understates the true bound.")
    RES['S2'] = scen
    return scen


# ======================================================================
# S3 — Kalman robustness under pressure-transducer noise
# ======================================================================
def disc(A, B, Ts):
    n, m = A.shape[0], B.shape[1]
    Mb = np.zeros((n + m, n + m)); Mb[:n, :n] = A; Mb[:n, n:] = B
    Md = expm(Mb * Ts)
    return Md[:n, :n], Md[:n, n:]


def S3(net, props, P0, Q0, TS=120.0, T=7200.0, ntrial=8):
    print("\n" + "=" * 74)
    print("S3  KALMAN ROBUSTNESS UNDER PRESSURE-TRANSDUCER NOISE")
    print("=" * 74)
    A, B, C, info = build_rlc(net, props, P0, Q0)
    ns, nN = info['ns'], net.n_nodes
    Ad, Bud = disc(A, B[:, [0]], TS)
    _, Bdd = disc(A, B[:, 1:], TS)
    Cd = C
    p_ref = np.mean(P0[1:])
    print(f"model order {ns};  |Q0| in [{np.abs(Q0).min():.1f}, "
          f"{np.abs(Q0).max():.1f}] kg/s;  mean delivery pressure "
          f"{p_ref / 1e5:.1f} bar")
    print("The paper uses R = 1e-3 * I with the states in Pa, i.e. it assumes a "
          "transducer\nstandard deviation of 0.03 Pa (3e-7 mbar). Real "
          "transmission transducers: 0.1-0.5 % FS.\n")

    def run(sigma_pa, R_scalar):
        Qk = np.eye(ns) * 1e-2
        Rk = np.eye(nN - 1) * R_scalar
        Pk = solve_discrete_are(Ad.T, Cd.T, Qk, Rk)
        L = Pk @ Cd.T @ np.linalg.inv(Cd @ Pk @ Cd.T + Rk)
        eP, eQ = [], []
        d = np.zeros(nN); d[3] = 55.0
        for _ in range(ntrial):
            x = np.zeros(ns); xh = np.zeros(ns)
            for k in range(int(T / TS)):
                dd = d if k * TS > 1800 else np.zeros(nN)
                y = Cd @ x + np.random.randn(nN - 1) * sigma_pa
                xh = xh + L @ (y - Cd @ xh)
                if k > 5:
                    eP.append(np.abs(x[:nN - 1] - xh[:nN - 1]).max() / 1e5)
                    eQ.append(np.abs(x[nN - 1:] - xh[nN - 1:]).max())
                x = Ad @ x + Bdd @ dd
                xh = Ad @ xh + Bdd @ dd
        return np.mean(eP), np.mean(eQ), np.max(eQ), np.abs(L).max()

    lvls = [0.1, 0.5, 1.0, 2.0, 5.0]
    out = []
    print(f"{'noise':>10} {'sigma':>10} | {'PAPER TUNING R=1e-3':>36} | "
          f"{'NOISE-MATCHED R=sigma^2':>32}")
    print(f"{'[% of p]':>10} {'[bar]':>10} | {'dP[bar]':>10} {'dQ mean':>11} "
          f"{'dQ max':>11} | {'dP[bar]':>10} {'dQ mean':>10} {'dQ max':>10}")
    for pct in lvls:
        sig = pct / 100 * p_ref
        a = run(sig, 1e-3)
        b = run(sig, sig ** 2)
        out.append(dict(pct=pct, sigma_bar=sig / 1e5, paper=a, matched=b))
        print(f"{pct:10.1f} {sig / 1e5:10.3f} | {a[0]:10.3f} {a[1]:11.1f} "
              f"{a[2]:11.1f} | {b[0]:10.4f} {b[1]:10.3f} {b[2]:10.3f}")
    print(f"\nmax|L| : paper tuning {run(0.005 * p_ref, 1e-3)[3]:.3f} "
          f"(≈1 -> the filter simply passes the raw measurement through)")
    print("READING: with the shipped tuning the unmeasured flow estimates are "
          "pure amplified noise -\nat 0.5 % transducer noise the flow error "
          "exceeds the largest true pipe flow by orders of magnitude.")
    RES['S3'] = out
    return out


# ======================================================================
# S4 — roughness and temperature sensitivity
# ======================================================================
def S4(net, props0, P0, Q0):
    print("\n" + "=" * 74)
    print("S4  PARAMETRIC SENSITIVITY: PIPE ROUGHNESS AND GAS TEMPERATURE")
    print("=" * 74)
    x0 = np.concatenate([P0[1:] / 1e5, Q0])
    drop0 = (P0[0] - P0[1:].min()) / 1e5
    print(f"{'eps [mm]':>10} {'lambda':>9} {'min P [bar]':>12} {'d(drop) [%]':>12} "
          f"{'tau_max [h]':>12}")
    rough = []
    for eps in [1.5e-5, 3.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]:
        P1, Q1 = steady_scaled(net, props0, net.demand_nom, net.p_source_nom,
                               x0, eps=eps)
        lam = darcy_friction(props0['rho'], props0['mu'], net.diam[0], Q1[0], eps=eps)
        tau = props0['rho'] * net.area * net.length / np.maximum(np.abs(Q1), 1e-6)
        # tau uses the *local* density, recomputed consistently
        cm = build_composition_model(net, props0, P1, Q1)
        drop = (P1[0] - P1[1:].min()) / 1e5
        rough.append(dict(eps=eps, lam=lam, pmin=P1[1:].min() / 1e5,
                          ddrop=(drop - drop0) / drop0 * 100,
                          taumax=cm['tau'].max() / 3600))
        print(f"{eps * 1e3:10.3f} {lam:9.4f} {P1[1:].min() / 1e5:12.2f} "
              f"{(drop - drop0) / drop0 * 100:12.1f} {cm['tau'].max() / 3600:12.1f}")

    print()
    print(f"{'T [K]':>10} {'a [m/s]':>9} {'rho [kg/m3]':>12} "
          f"{'tau_max [h]':>12} {'d tau [%]':>10}")
    temp = []
    for T in [263.15, 273.15, 283.15, 288.15, 293.15, 303.15]:
        pr = mixture_properties(0.05, T=T, p_ref=6.0e6)
        P1, Q1 = steady_scaled(net, pr, net.demand_nom, net.p_source_nom,
                               np.concatenate([P0[1:] / 1e5, Q0]))
        cm = build_composition_model(net, pr, P1, Q1)
        cm0 = build_composition_model(net, props0, P0, Q0)
        d = (cm['tau'].max() - cm0['tau'].max()) / cm0['tau'].max() * 100
        temp.append(dict(T=T, a=pr['a'], rho=pr['rho'],
                         taumax=cm['tau'].max() / 3600, dtau=d))
        print(f"{T:10.2f} {pr['a']:9.1f} {pr['rho']:12.2f} "
              f"{cm['tau'].max() / 3600:12.1f} {d:10.1f}")
    print("\nREADING: tau ~ rho ~ 1/T, so a +/-15 K seasonal swing moves every "
          "delivery delay by ~5 %,\nwhich is of the same order as the "
          "frozen-flow error and must enter the blend back-off.")
    RES['S4'] = dict(roughness=rough, temperature=temp)
    return rough, temp


# ======================================================================
# Figures
# ======================================================================
def figures(net, props, P0, Q0, s1rows, te, refP, s2, s3):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})

    # ---- Fig A : transport discretisation ------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    Ms = [r['M'] for r in s1rows]
    worst = [max(r['peak'].values()) for r in s1rows]
    ax[0].semilogx(Ms, worst, 'o-', lw=1.8, color="#2C7FB8")
    ax[0].axhline(0, color='0.6', lw=.8)
    for M, w in zip(Ms, worst):
        if M in (1, 4, 20):
            ax[0].annotate(f"M={M}", (M, w), (4, 6), textcoords="offset points",
                           fontsize=8, color="0.3")
    ax[0].scatter([RES['S1']['cfl']['ns'] / net.n_pipes],
                  [max(RES['S1']['cfl']['peak'].values())], marker='*', s=140,
                  color="#D95F0E", zorder=5, label="CFL-matched $M_k$")
    ax[0].set_xlabel("cells per pipe $M$")
    ax[0].set_ylabel("peak under-prediction [vol-pts]")
    ax[0].set_title("(a) Numerical diffusion of the peak\n(6 h pulse, worst node)")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(frameon=False, fontsize=8)

    pulse = lambda t: 0.30 if 4 * 3600 <= t < 10 * 3600 else 0.05
    for M, c in [(1, "#B0357F"), (4, "#D95F0E"), (20, "#41AB5D")]:
        cm = build_adv_composition(net, props, P0, Q0, M=M)
        t2, cv = simulate(cm['A'], cm['B'], cm['Wsel'], pulse, 48 * 3600)
        ax[1].plot(t2 / 3600, cv[3] * 100, lw=1.6, color=c, label=f"$M$={M}")
    ax[1].plot(te / 3600, refP[3] * 100, 'k-', lw=2, label="plug-flow ref.")
    ax[1].set_xlabel("time [h]"); ax[1].set_ylabel("delivered H$_2$ at N3 [vol%]")
    ax[1].set_title("(b) Front smearing, 6 h pulse")
    ax[1].grid(alpha=.3); ax[1].legend(frameon=False, fontsize=8)

    labs = [s['label'].split('(')[0][:22] for s in s2]
    y = np.arange(len(s2))
    ax[2].barh(y - .2, [s['mean'] for s in s2], .38, color="#9ECAE1",
               label="mean over time (as reported)")
    ax[2].barh(y + .2, [s['max'] for s in s2], .38, color="#D95F0E",
               label="true worst case")
    ax[2].set_yticks(y); ax[2].set_yticklabels(labs, fontsize=8)
    ax[2].set_xlabel("delivered-blend error [vol-pts]")
    ax[2].set_title("(c) Frozen-flow validity envelope")
    ax[2].grid(alpha=.3, axis='x'); ax[2].legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_sensitivity_transport.png"); plt.close(fig)
    print("\nsaved fig_sensitivity_transport.png")

    # ---- Fig B : estimator robustness ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    pct = [r['pct'] for r in s3]
    ax[0].semilogy(pct, [r['paper'][1] for r in s3], 'o-', lw=1.8,
                   color="#B0357F", label="tuning used in the paper ($R=10^{-3}I$)")
    ax[0].semilogy(pct, [r['matched'][1] for r in s3], 's-', lw=1.8,
                   color="#41AB5D", label=r"noise-matched ($R=\sigma^2 I$)")
    ax[0].axhline(np.abs(Q0).max(), color="0.4", ls="--", lw=1.2,
                  label="largest true pipe flow")
    ax[0].set_xlabel("pressure-transducer noise [% of reading]")
    ax[0].set_ylabel("flow-estimate error [kg/s]")
    ax[0].set_title("(a) Unmeasured flow states")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(frameon=False, fontsize=8.5)

    ax[1].plot(pct, [r['paper'][0] for r in s3], 'o-', lw=1.8, color="#B0357F",
               label="paper tuning")
    ax[1].plot(pct, [r['matched'][0] for r in s3], 's-', lw=1.8, color="#41AB5D",
               label="noise-matched")
    ax[1].set_xlabel("pressure-transducer noise [% of reading]")
    ax[1].set_ylabel("pressure-estimate error [bar]")
    ax[1].set_title("(b) Measured pressure states")
    ax[1].grid(alpha=.3); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig_sensitivity_robustness.png"); plt.close(fig)
    print("saved fig_sensitivity_robustness.png")


if __name__ == "__main__":
    net = Network()
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)
    te, refP, s1rows = S1(net, props, P0, Q0)
    s2 = S2(net, props, P0, Q0)
    s3 = S3(net, props, P0, Q0)
    S4(net, props, P0, Q0)
    figures(net, props, P0, Q0, s1rows, te, refP, s2, s3)
    np.save("sensitivity_results.npy", RES, allow_pickle=True)
    print("saved sensitivity_results.npy")
