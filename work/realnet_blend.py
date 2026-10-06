"""realnet_blend.py - Case Study 3: closed-loop TWO-POINT blend scheduling on the
real SciGRID_gas German H-gas sub-network.

IMPLEMENTATION NOTES
--------------------
(1) PREDICTOR.  The controller no longer uses a coarse M = 4 upwind state-space.
    It uses the finite-impulse-response (Markov-parameter) representation of a
    FINE transport model (M = 60 cells per pipe), built offline by
    `transport_fir.markov_params`.  Because the transport is linear and
    time-invariant at a frozen flow field, this is an exact re-parameterisation:
    prediction accuracy is decoupled from online cost, and the optimisation
    carries no transport states at all - only the H injection decisions.

(2) PLANT.  The plant is now INDEPENDENT of the predictor:
      * a fine upwind model (M = 30) rather than the controller's model;
      * a time-varying diurnal demand profile, with the steady flow field, the
        residence times and the junction mixing weights RECOMPUTED as the
        demands change - so the frozen-flow assumption of the predictor is
        genuinely violated;
      * pipe reversal handled explicitly (the gas column in a reversing pipe is
        flipped, and the upwind structure rebuilt);
      * injection actuator noise on the realised set-point.
    Advancing the "plant" with the controller's own discrete model would
    make zero constraint violation a tautology, so it is avoided here.

(3) BACK-OFF.  The delivered-composition constraint is imposed at
    y_max - DELTA, with DELTA taken from the quantified transport-model error
    envelope, mirroring the 3.5 bar back-off already used in the pressure loop.

(4) HORIZON AND RUN LENGTH.  Both are now derived from the network's own
    residence-time structure (tau_max = 81 h on this sub-network) rather than
    assumed: the FIR tap count covers 3 tau_max and the simulation continues
    until every launched front has been delivered.
"""
import numpy as np
import time
import cvxpy as cp

from gas_properties import mixture_properties
from network import steady_state, steady_continuation
from composition import build_composition_model
from transport_fir import (build_upwind, markov_params, fir_settling,
                           discretise, fir_matrices)
import realnet

# ---------------------------------------------------------------- settings
TS = 1800.0              # blend-loop sampling [s]
Y_LIMIT = 0.20           # interchangeability cap on delivered blend
Y_BASE = 0.05            # baseline blend always injectable
DELTA = 0.02             # constraint back-off [vol frac] - see backoff_study()
M_PRED = 60              # cells/pipe of the model the FIR is built from
M_PLANT = 30             # cells/pipe of the independent plant
T_INJ = 24 * 3600.0      # green-H2 window
SIG_ACT = 0.005          # injection actuator noise [vol frac, 1 sigma]
DU_MAX = 0.05            # injector ramp limit [vol frac per 30 min step]
RETUNE_TOL = 0.02        # rebuild plant discretisation when tau drifts > 2 %


def avail_east(t):
    """Eastern midday solar surplus."""
    h = t / 3600.0
    if 9 <= h < 16:
        return 0.30
    if 16 <= h < 18:
        return 0.20
    return Y_BASE


def avail_west(t):
    """Western afternoon-evening wind."""
    h = t / 3600.0
    if 15 <= h < 22:
        return 0.28
    return Y_BASE


_PHASE = None


def set_phase_seed(seed):
    """Redraw the per-node demand phases. Used by the ensemble study, which
    repeats the closed-loop run over independent draws of the two random inputs
    the scenario carries: the demand phases and the injector actuator noise."""
    global _PHASE
    _PHASE = np.random.default_rng(seed).uniform(0, 2 * np.pi, 22)


def demand_profile(net, t, amp=0.15):
    """Diurnal demand variation with a NODE-DEPENDENT phase.

    Real offtakes do not peak together: residential, industrial and
    power-station loads have different daily shapes. A uniform scaling would
    leave the flow *split* between parallel loop paths unchanged, so it would
    not test the frozen-flow assumption at all. Distinct per-node phases move
    the loop chords - which is what makes the residence times, the junction
    mixing weights and, at large amplitude, the flow directions time-varying.
    """
    global _PHASE
    if _PHASE is None or len(_PHASE) != net.n_nodes:
        rng = np.random.default_rng(7)
        _PHASE = rng.uniform(0, 2 * np.pi, net.n_nodes)
    h = t / 3600.0
    f = 1.0 + amp * np.sin(2 * np.pi * h / 24.0 + _PHASE)
    d = net.demand_nom * f
    d[net.source] = 0.0
    return d


MH2 = mixture_properties(1.0)['M']
MNG = mixture_properties(0.0)['M']


def wmass(y):
    """Hydrogen mass fraction from hydrogen volume (mole) fraction."""
    return y * MH2 / (y * MH2 + (1 - y) * MNG)


# ---------------------------------------------------------------- plant
class Plant:
    """Fine transport model on a time-varying flow field."""

    def __init__(self, net, props, P0, Q0, inj_nodes, M=M_PLANT, seed=0):
        self.net, self.props = net, props
        self.inj = inj_nodes
        self.M = M
        self.rng = np.random.default_rng(seed)
        self.P, self.Q = P0.copy(), Q0.copy()
        self.mdl = build_upwind(net, props, P0, Q0, np.full(net.n_pipes, M),
                                inj_nodes)
        self.tau_ref = self.mdl['tau'].copy()
        self.tau_nom = self.mdl['tau'].copy()
        self.tau_drift = 0.0
        self.sgn = np.sign(self.Q)
        self.Ad, self.Bd = discretise(self.mdl['A'], self.mdl['B'], TS)
        self.y = np.full(self.mdl['nstate'], Y_BASE)
        self.n_rebuild = 1
        self.n_reversal = 0
        # The demand profile is exactly 24 h periodic, so the flow field - and
        # therefore the discretised transport operator - repeats each day. Cache
        # by time-of-day slot so the run cost does not grow with its length.
        self._cache = {}

    def _flip_reversed(self, new_sgn):
        """A pipe whose flow has reversed carries the same gas column traversed
        in the opposite direction: flip that pipe's cell block."""
        off, Mk = self.mdl['offsets'], self.mdl['Mk']
        flipped = 0
        for k in range(self.net.n_pipes):
            if new_sgn[k] != self.sgn[k]:
                sl = slice(off[k], off[k] + Mk[k])
                self.y[sl] = self.y[sl][::-1]
                flipped += 1
        self.n_reversal += flipped
        self.sgn = new_sgn

    def step(self, t, u_cmd, u_avail):
        # --- realised injection: actuator noise, but never more green hydrogen
        #     than is physically available, and never less than the base blend
        u = np.clip(u_cmd + self.rng.normal(0, SIG_ACT, size=len(u_cmd)),
                    Y_BASE, np.asarray(u_avail, dtype=float))
        # --- current flow field from the current demands
        dem = demand_profile(self.net, t)
        try:
            self.P, self.Q = steady_state(self.net, self.props, demand=dem,
                                          x0=np.concatenate([self.P[1:] / 1e5,
                                                             self.Q]))
        except RuntimeError:
            self.P, self.Q = steady_continuation(self.net, self.props, dem)
        new_sgn = np.sign(self.Q)
        reversed_now = np.any(new_sgn != self.sgn)
        if reversed_now:
            self._flip_reversed(new_sgn)
        # --- rebuild transport structure if the flow field has drifted
        cm = build_composition_model(self.net, self.props, self.P, self.Q)
        self.tau_drift = max(getattr(self, 'tau_drift', 0.0),
                             float(np.max(np.abs(cm['tau'] - self.tau_nom) /
                                          self.tau_nom)))
        slot = int(round((t % (24 * 3600.0)) / TS))
        if slot not in self._cache:
            mdl = build_upwind(self.net, self.props, self.P, self.Q,
                               np.full(self.net.n_pipes, self.M), self.inj)
            self._cache[slot] = (mdl, ) + discretise(mdl['A'], mdl['B'], TS)
            self.n_rebuild += 1
        self.mdl, self.Ad, self.Bd = self._cache[slot]
        # --- advance one sampling interval
        self.y = self.Ad @ self.y + self.Bd @ u
        return u, self.mdl['Wsel'] @ self.y


# ---------------------------------------------------------------- controller
def build_controller(G, Hu, Hp, n_inj, n_del, y_lim):
    """Dual-horizon predictive scheduler over the injection moves only.

    Hu : CONTROL horizon - the number of injection moves optimised.
    Hp : PREDICTION / CONSTRAINT horizon - long enough to cover the network's
         slowest delivery delay, so that a front committed now cannot breach the
         limit after the optimisation window has closed. Beyond Hu the injection
         is held at the baseline blend, which is what physically happens once
         the green-hydrogen window ends.

    Because the FIR representation carries no transport states, the decision
    vector is 2*Hu regardless of how long Hp is: the long constraint horizon is
    essentially free.
    """
    Phi_full, Psi = fir_matrices(G, Hp)          # (Hp*n_del) x (Hp*n_inj)
    Phi = Phi_full[:, :Hu * n_inj]               # only the first Hu moves vary
    du = cp.Variable((Hu, n_inj))
    free = cp.Parameter(Hp * n_del)              # committed-history response
    aE = cp.Parameter(Hu, nonneg=True)
    aW = cp.Parameter(Hu, nonneg=True)
    uprev = cp.Parameter(n_inj)
    wgt = cp.Parameter(n_inj, nonneg=True)
    s = cp.Variable(Hp, nonneg=True)          # one slack per prediction step
    # expand the per-step slack across the n_del constrained nodes
    Exp = np.repeat(np.eye(Hp), n_del, axis=0)

    z = Phi @ cp.vec(du, order='C') + free + Y_BASE
    cons = [z <= y_lim + Exp @ s,
            du[:, 0] >= 0, du[:, 0] <= aE - Y_BASE,
            du[:, 1] >= 0, du[:, 1] <= aW - Y_BASE,
            # Physical ramp limit of the injection station, imposed as a hard
            # constraint rather than absorbed into the smoothness weight, so the
            # schedule is realisable by the equipment. Only the UPWARD rate is
            # limited: an injector can always be closed quickly, and bounding the
            # downward rate as well would make the problem infeasible whenever
            # the renewable availability collapses within one step.
            cp.diff(du, axis=0) <= DU_MAX,
            du[0] - (uprev - Y_BASE) <= DU_MAX]
    # Objective scaling: the throughput-weighted uptake is summed over the
    # horizon (not averaged), so that the benefit of injecting is O(Hu) and is
    # not swamped by the O(1) smoothness penalty on a single move.
    cost = (-cp.sum(du @ wgt) / cp.max(wgt) + 1e5 * cp.sum(s)
            + 1.0 * cp.sum_squares(du[0] - (uprev - Y_BASE))
            + 1.0 * cp.sum_squares(cp.diff(du, axis=0)))
    prob = cp.Problem(cp.Minimize(cost), cons)
    return prob, du, free, aE, aW, uprev, wgt, Phi, Psi


def run(controller="mpc", Hu=None, Hp=None, N=None, T_END=None, verbose=True,
        nominal_plant=False, fc_east=None, fc_west=None,
        phase_seed=None, act_seed=0):
    """fc_east / fc_west are the availability profiles the SCHEDULER plans
    against. They default to the true profiles, which is the perfect-forecast
    case. Passing a perturbed pair makes the scheduler plan against a forecast
    while the plant continues to clip the realised injection to the true
    availability, which is the only way a forecast error can be tested: if the
    same perturbed profile drives both, the scheduler simply has perfect
    knowledge of a different day."""
    """nominal_plant=True reproduces the PREVIOUS setup, in which the plant was
    advanced with the controller's own model on the nominal flow field. It is
    retained only so that the change in reported uptake between revisions can be
    attributed: it is not a closed-loop test."""
    if phase_seed is not None:
        set_phase_seed(phase_seed)
    net, props, P0, Q0 = realnet.build()
    src, hub = net.source, net.western_hub
    inj = [src, hub]
    deliv = [n for n in range(net.n_nodes) if n != src]
    n_del, n_inj = len(deliv), 2

    # ---- offline: FIR from a fine transport model at the nominal flow field
    fine = build_upwind(net, props, P0, Q0, np.full(net.n_pipes, M_PRED), inj)
    cm = build_composition_model(net, props, P0, Q0)
    tau_max_h = cm['tau'].max() / 3600
    if N is None:
        N = int(np.ceil(3.0 * cm['tau'].max() / TS))
    G = markov_params(fine['A'], fine['B'], fine['Wsel'], TS, N, rows=deliv)
    n_set = fir_settling(G)
    if Hu is None:
        Hu = int(np.ceil(T_INJ / TS))          # control horizon = H2 window
    if Hp is None:
        Hp = int(np.ceil(n_set * 1.05))        # constraint horizon = settling
    if T_END is None:
        T_END = T_INJ + N * TS
    gsum = G.sum(axis=0).sum(axis=1)           # total gain, all injectors
    if verbose:
        print(f"  tau_max = {tau_max_h:.1f} h;  FIR taps N = {N} "
              f"({N*TS/3600:.0f} h);  99.9 % settling at {n_set} taps "
              f"({n_set*TS/3600:.0f} h)")
        print(f"  control horizon Hu = {Hu} ({Hu*TS/3600:.0f} h);  "
              f"constraint horizon Hp = {Hp} ({Hp*TS/3600:.0f} h);  "
              f"run length {T_END/3600:.0f} h")
        print(f"  predictor: {fine['nstate']} states offline -> "
              f"{Hu*n_inj} decision variables online")
        print(f"  FIR steady gain per node (mass conservation check): "
              f"min {gsum.min():.6f}  max {gsum.max():.6f}  (exact value 1)")
        print(f"  nodes reachable from the western injector: "
              f"{int((G.sum(axis=0)[:,1] > 1e-6).sum())} of {n_del}")

    # injection throughputs for the mass-weighted uptake objective
    up = cm['upstream']
    MDOT_E = sum(abs(Q0[k]) for k in range(net.n_pipes) if up[k] == src)
    MDOT_W = sum(abs(Q0[k]) for k in range(net.n_pipes) if up[k] == hub)

    y_lim = Y_LIMIT - DELTA if controller == "mpc" else Y_LIMIT
    if controller == "mpc":
        (prob, v_du, p_free, p_aE, p_aW, p_uprev, p_wgt,
         Phi, Psi) = build_controller(G, Hu, Hp, n_inj, n_del, y_lim)
        p_wgt.value = np.array([MDOT_E, MDOT_W])

    if nominal_plant:
        class _Nominal:
            def __init__(self):
                a = build_upwind(net, props, P0, Q0,
                                 np.full(net.n_pipes, M_PRED), inj)
                self.Ad, self.Bd = discretise(a['A'], a['B'], TS)
                self.W = a['Wsel']; self.y = np.full(a['nstate'], Y_BASE)
                self.n_rebuild = 1; self.n_reversal = 0; self.tau_drift = 0.0
            def step(self, t, u_cmd, u_avail):
                u = np.asarray(u_cmd, dtype=float)
                self.y = self.Ad @ self.y + self.Bd @ u
                return u, self.W @ self.y
        plant = _Nominal()
    else:
        plant = Plant(net, props, P0, Q0, inj, seed=act_seed)
    if fc_east is None:
        fc_east = avail_east
    if fc_west is None:
        fc_west = avail_west
    hist = np.full((N, n_inj), Y_BASE)     # hist[0] = most recent
    u_last = np.full(n_inj, Y_BASE)
    nsteps = int(T_END / TS)
    tl, uEl, uWl, ydl, aEl, aWl, sm = [], [], [], [], [], [], []

    for stp in range(nsteps):
        t = stp * TS
        aE_now = avail_east(t) if t < T_INJ else Y_BASE
        aW_now = avail_west(t) if t < T_INJ else Y_BASE
        if t < T_INJ and controller == "mpc":
            # the horizon is the FORECAST; aE_now/aW_now below are the truth
            p_aE.value = np.array([fc_east(t + j*TS) if (t + j*TS) < T_INJ
                                   else Y_BASE for j in range(Hu)])
            p_aW.value = np.array([fc_west(t + j*TS) if (t + j*TS) < T_INJ
                                   else Y_BASE for j in range(Hu)])
            p_free.value = (Psi @ (hist - Y_BASE).reshape(-1))
            p_uprev.value = u_last
            t0 = time.perf_counter()
            prob.solve(solver=cp.CLARABEL, warm_start=True)
            sm.append((time.perf_counter() - t0) * 1000)
            u_cmd = Y_BASE + np.asarray(v_du.value[0]).ravel()
        elif t < T_INJ:                    # composition-blind full injection
            u_cmd = np.array([aE_now, aW_now])
        else:
            u_cmd = np.full(n_inj, Y_BASE)

        u_real, zdel = plant.step(t, u_cmd, [aE_now, aW_now])
        hist = np.vstack([u_real, hist[:-1]])
        u_last = u_real
        tl.append((t + TS) / 3600)
        uEl.append(u_real[0]); uWl.append(u_real[1])
        ydl.append(zdel[deliv]); aEl.append(aE_now); aWl.append(aW_now)

    return dict(t=np.array(tl), uE=np.array(uEl), uW=np.array(uWl),
                ydel=np.array(ydl), aE=np.array(aEl), aW=np.array(aWl),
                MDOT_E=MDOT_E, MDOT_W=MDOT_W, Hu=Hu, Hp=Hp, N=N, T_END=T_END,
                tau_max_h=tau_max_h, y_lim=y_lim, ndeliv=n_del,
                n_rebuild=plant.n_rebuild, n_reversal=plant.n_reversal,
                tau_drift=plant.tau_drift,
                solve_ms=float(np.median(sm)) if sm else np.nan)


def uptake(r):
    ts = r['t'] * 3600

    def mass(u, a, MD):
        ab = np.trapezoid(np.maximum(wmass(u) - wmass(Y_BASE), 0) * MD, ts)
        av = np.trapezoid(np.maximum(wmass(a) - wmass(Y_BASE), 0) * MD, ts)
        return ab, av

    abE, avE = mass(r['uE'], r['aE'], r['MDOT_E'])
    abW, avW = mass(r['uW'], r['aW'], r['MDOT_W'])

    def vfrac(u, a):
        return (np.trapezoid(np.maximum(u - Y_BASE, 0), ts) /
                max(np.trapezoid(np.maximum(a - Y_BASE, 0), ts), 1e-9))

    volE, volW = vfrac(r['uE'], r['aE']), vfrac(r['uW'], r['aW'])
    worst = r['ydel'].max(axis=1)
    post = r['t'] > T_INJ / 3600
    return dict(mass=(abE + abW) / (avE + avW) * 100,
                vol=(r['MDOT_E']*volE + r['MDOT_W']*volW) /
                    (r['MDOT_E'] + r['MDOT_W']) * 100,
                peak=worst.max() * 100,
                viol=(worst > Y_LIMIT + 1e-4).mean() * 100,
                margin=(Y_LIMIT - worst.max()) * 100,
                peak_post=r['ydel'][post].max() * 100 if post.any() else 0.0)


if __name__ == "__main__":
    print("=== CS3 two-point blend control ===")
    print(f"predictor: FIR of an M={M_PRED} transport model (no online states); "
          f"plant: independent M={M_PLANT} model on a time-varying flow field")
    print(f"back-off:  delivered constraint imposed at "
          f"{(Y_LIMIT-DELTA)*100:.0f} vol% against a {Y_LIMIT*100:.0f} vol% limit\n")
    mpc = run("mpc")
    rea = run("reactive", Hu=mpc['Hu'], Hp=mpc['Hp'], N=mpc['N'],
              T_END=mpc['T_END'], verbose=False)
    print(f"\ninjection throughputs: east {mpc['MDOT_E']:.0f} kg/s, "
          f"west(hub) {mpc['MDOT_W']:.0f} kg/s | {mpc['ndeliv']} customer "
          f"nodes constrained")
    print(f"plant: {mpc['n_rebuild']} distinct transport operators over the "
          f"diurnal cycle; peak residence-time drift from nominal "
          f"{mpc['tau_drift']*100:.1f} %; {mpc['n_reversal']} pipe-reversal "
          f"events\n")
    for nm, r in [("MPC (two-point, FIR)", mpc),
                  ("Full-injection baseline", rea)]:
        m = uptake(r)
        print(f"{nm}")
        print(f"   peak delivered {m['peak']:.2f} vol%  (margin to limit "
              f"{m['margin']:+.2f} pts) | uptake vol {m['vol']:.1f}% / "
              f"MASS {m['mass']:.1f}%")
        print(f"   time in violation {m['viol']:.1f}%  | peak after injection "
              f"window {m['peak_post']:.2f} vol%"
              + (f"  | median solve {r['solve_ms']:.0f} ms"
                 if nm.startswith('MPC') else ""))
    np.save("realnet_blend_results.npy", dict(mpc=mpc, rea=rea),
            allow_pickle=True)
    print("\nsaved realnet_blend_results.npy")
