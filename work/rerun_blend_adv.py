"""rerun_blend_adv.py - didactic-network blend control, rebuilt on the same
basis as Case Study 3 (see realnet_blend.py for the full rationale):

  * predictor  : FIR of a fine (M = 60) transport model - no online states;
  * plant      : independent fine (M = 30) model on a TIME-VARYING flow field,
                 with injection actuator noise, so the frozen-flow assumption is
                 genuinely violated;
  * constraint : imposed at y_max - DELTA, with DELTA from the quantified
                 transport-error envelope (structural_analysis.py, S3);
  * horizons   : control horizon = injection window, constraint horizon = the
                 network's own 99.9 % settling time.
"""
import time
import numpy as np
import cvxpy as cp

from gas_properties import mixture_properties
from network import Network, steady_state
from composition import build_composition_model
from transport_fir import (build_upwind, markov_params, fir_settling,
                           discretise, fir_matrices)

TS = 1800.0
Y_LIMIT, Y_BASE, DELTA = 0.20, 0.05, 0.02
M_PRED, M_PLANT = 60, 30
T_INJ = 24 * 3600.0
SIG_ACT = 0.005
DU_MAX = 0.05            # injector ramp limit [vol frac per 30 min step]
DELIV = [1, 2, 3, 4, 5]
MDOT = 195.0                      # total gas throughput of the didactic network

MH2 = mixture_properties(1.0)['M']
MNG = mixture_properties(0.0)['M']
wmass = lambda y: y * MH2 / (y * MH2 + (1 - y) * MNG)

_PHASE = None


def availability(t):
    h = t / 3600.0
    if 8 <= h < 17:
        return 0.30
    if 19 <= h < 21:
        return 0.26
    return Y_BASE


def demand_profile(net, t, amp=0.15):
    global _PHASE
    if _PHASE is None:
        _PHASE = np.random.default_rng(3).uniform(0, 2 * np.pi, net.n_nodes)
    d = net.demand_nom * (1.0 + amp * np.sin(2 * np.pi * t / 86400.0 + _PHASE))
    d[net.source] = 0.0
    return d


class Plant:
    def __init__(self, net, props, P0, Q0, seed=0):
        self.net, self.props, self.rng = net, props, np.random.default_rng(seed)
        self.P, self.Q = P0.copy(), Q0.copy()
        self.mdl = build_upwind(net, props, P0, Q0,
                                np.full(net.n_pipes, M_PLANT))
        self.tau_nom = self.mdl['tau'].copy()
        self.tau_drift = 0.0
        self.sgn = np.sign(Q0)
        self.Ad, self.Bd = discretise(self.mdl['A'], self.mdl['B'], TS)
        self.y = np.full(self.mdl['nstate'], Y_BASE)
        self._cache, self.n_rebuild, self.n_reversal = {}, 1, 0

    def step(self, t, u_cmd, u_avail):
        u = float(np.clip(u_cmd + self.rng.normal(0, SIG_ACT, 1),
                          Y_BASE, u_avail)[0])
        dem = demand_profile(self.net, t)
        self.P, self.Q = steady_state(self.net, self.props, demand=dem,
                                      x0=np.concatenate([self.P[1:]/1e5, self.Q]))
        s = np.sign(self.Q)
        if np.any(s != self.sgn):
            off, Mk = self.mdl['offsets'], self.mdl['Mk']
            for k in range(self.net.n_pipes):
                if s[k] != self.sgn[k]:
                    sl = slice(off[k], off[k] + Mk[k])
                    self.y[sl] = self.y[sl][::-1]
                    self.n_reversal += 1
            self.sgn = s
            self._cache.clear()
        cm = build_composition_model(self.net, self.props, self.P, self.Q)
        self.tau_drift = max(self.tau_drift,
                             float(np.max(np.abs(cm['tau'] - self.tau_nom)
                                          / self.tau_nom)))
        slot = int(round((t % 86400.0) / TS))
        if slot not in self._cache:
            m = build_upwind(self.net, self.props, self.P, self.Q,
                             np.full(self.net.n_pipes, M_PLANT))
            self._cache[slot] = (m,) + discretise(m['A'], m['B'], TS)
            self.n_rebuild += 1
        self.mdl, self.Ad, self.Bd = self._cache[slot]
        self.y = self.Ad @ self.y + self.Bd[:, 0] * u
        return u, self.mdl["Wsel"] @ self.y


def run(controller="mpc", Hu=None, Hp=None, N=None, T_END=None, verbose=True):
    net = Network()
    props = mixture_properties(Y_BASE)
    P0, Q0 = steady_state(net, props)
    cm = build_composition_model(net, props, P0, Q0)
    if N is None:
        N = int(np.ceil(3.0 * cm['tau'].max() / TS))
    fine = build_upwind(net, props, P0, Q0, np.full(net.n_pipes, M_PRED))
    G = markov_params(fine['A'], fine['B'], fine['Wsel'], TS, N, rows=DELIV)
    n_set = fir_settling(G)
    Hu = Hu or int(np.ceil(T_INJ / TS))
    Hp = Hp or int(np.ceil(n_set * 1.05))
    T_END = T_END or (T_INJ + N * TS)
    if verbose:
        print(f"  tau_max {cm['tau'].max()/3600:.1f} h | FIR taps {N} "
              f"({N*TS/3600:.0f} h), 99.9 % settling {n_set} "
              f"({n_set*TS/3600:.0f} h)")
        print(f"  Hu {Hu} ({Hu*TS/3600:.0f} h) | Hp {Hp} ({Hp*TS/3600:.0f} h) "
              f"| run {T_END/3600:.0f} h | {Hu} decision variables")
        print(f"  FIR steady gain per node "
              f"{G.sum(axis=0)[:,0].min():.6f}..{G.sum(axis=0)[:,0].max():.6f} "
              f"(mass conservation)")

    y_lim = Y_LIMIT - DELTA if controller == "mpc" else Y_LIMIT
    if controller == "mpc":
        Phi_f, Psi = fir_matrices(G, Hp)
        Phi = Phi_f[:, :Hu]
        du = cp.Variable(Hu)
        free = cp.Parameter(Hp * len(DELIV))
        av = cp.Parameter(Hu, nonneg=True)
        uprev = cp.Parameter()
        s = cp.Variable(Hp, nonneg=True)
        Exp = np.repeat(np.eye(Hp), len(DELIV), axis=0)
        z = Phi @ du + free + Y_BASE
        cons = [z <= y_lim + Exp @ s, du >= 0, du <= av - Y_BASE,
                cp.diff(du) <= DU_MAX,
                du[0] - (uprev - Y_BASE) <= DU_MAX]
        # Objective scaling. The uptake term must be the SUM over the horizon,
        # not its mean: with a mean the benefit of injecting is O(1/Hu) while a
        # single injection move costs O(1) in the smoothness term, so the move
        # penalty silently dominates and the scheduler under-injects.
        cost = (-cp.sum(du) + 1e5 * cp.sum(s)
                + 1.0 * cp.square(du[0] - (uprev - Y_BASE))
                + 1.0 * cp.sum_squares(cp.diff(du)))
        prob = cp.Problem(cp.Minimize(cost), cons)

    plant = Plant(net, props, P0, Q0)
    hist = np.full((N, 1), Y_BASE)
    u_last = Y_BASE
    tl, ul, ydl, avl, sm = [], [], [], [], []
    for stp in range(int(T_END / TS)):
        t = stp * TS
        a_now = availability(t) if t < T_INJ else Y_BASE
        if t < T_INJ and controller == "mpc":
            av.value = np.array([availability(t + j*TS) if (t + j*TS) < T_INJ
                                 else Y_BASE for j in range(Hu)])
            free.value = Psi @ (hist - Y_BASE).reshape(-1)
            uprev.value = u_last
            t0 = time.perf_counter()
            prob.solve(solver=cp.CLARABEL, warm_start=True)
            sm.append((time.perf_counter() - t0) * 1000)
            u_cmd = Y_BASE + float(du.value[0])
        elif t < T_INJ:
            u_cmd = a_now
        else:
            u_cmd = Y_BASE
        u_real, z = plant.step(t, np.array([u_cmd]), np.array([a_now]))
        hist = np.vstack([[u_real], hist[:-1]])
        u_last = u_real
        tl.append((t + TS) / 3600); ul.append(u_real)
        ydl.append(z[DELIV]); avl.append(a_now)
    return dict(t=np.array(tl), u=np.array(ul), ydel=np.array(ydl),
                avail=np.array(avl), Hu=Hu, Hp=Hp, N=N, T_END=T_END,
                y_lim=y_lim, tau_drift=plant.tau_drift,
                n_reversal=plant.n_reversal,
                solve_ms=float(np.median(sm)) if sm else np.nan)


def uptake(r):
    ts = r['t'] * 3600
    u, av = r['u'], r['avail']
    vol = (np.trapezoid(np.maximum(u - Y_BASE, 0), ts) /
           np.trapezoid(np.maximum(av - Y_BASE, 0), ts) * 100)
    gi = np.trapezoid(np.maximum(wmass(u) - wmass(Y_BASE), 0) * MDOT, ts)
    ga = np.trapezoid(np.maximum(wmass(av) - wmass(Y_BASE), 0) * MDOT, ts)
    w = r['ydel'].max(axis=1)
    post = r['t'] > T_INJ / 3600
    return dict(vol=vol, mass=gi / ga * 100, peak=w.max() * 100,
                margin=(Y_LIMIT - w.max()) * 100,
                viol=(w > Y_LIMIT + 1e-4).mean() * 100,
                peak_post=r['ydel'][post].max() * 100 if post.any() else 0.0)


if __name__ == "__main__":
    print("=== Didactic-network blend control (FIR predictor, independent "
          "plant) ===")
    print(f"constraint imposed at {(Y_LIMIT-DELTA)*100:.0f} vol% against a "
          f"{Y_LIMIT*100:.0f} vol% interchangeability limit\n")
    mpc = run("mpc")
    rea = run("reactive", Hu=mpc['Hu'], Hp=mpc['Hp'], N=mpc['N'],
              T_END=mpc['T_END'], verbose=False)
    print(f"  plant: peak residence-time drift {mpc['tau_drift']*100:.1f} %, "
          f"{mpc['n_reversal']} reversal events\n")
    for nm, r in [("MPC", mpc), ("Full-injection baseline", rea)]:
        m = uptake(r)
        print(f"{nm}: peak {m['peak']:.2f} vol% (margin {m['margin']:+.2f} pts) "
              f"| uptake vol {m['vol']:.1f}% / MASS {m['mass']:.1f}% | "
              f"violation {m['viol']:.1f}% | post-window peak "
              f"{m['peak_post']:.2f}%"
              + (f" | median solve {r['solve_ms']:.0f} ms" if nm == 'MPC'
                 else ""))
    np.save("blend_adv_results.npy", dict(mpc=mpc, rea=rea), allow_pickle=True)
    print("\nsaved blend_adv_results.npy")
