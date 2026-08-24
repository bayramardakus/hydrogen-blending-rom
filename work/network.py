"""
network.py
----------
Representative meshed transmission sub-network (parameters grounded in the
SciGRID_gas European data model, Pluta et al. 2022) plus two models:

  (1) build_reference(): high-fidelity 1D transient model - a fine finite-volume
      (staggered) discretisation of the isothermal Euler equations with
      Darcy-Weisbach friction, per pipe. This is the CFD-proxy reference.

  (2) build_rlc(): reduced-order lumped RLC state-space (A,B,C). One flow state
      per pipe (inductance+resistance) and one pressure state per node
      (capacitance), assembled by nodal analysis (Kirchhoff's laws).

Isothermal governing PDE per pipe (mass flow phi [kg/s], pressure p [Pa]):
    dp/dt   = -(a^2/A) dphi/dx
    dphi/dt = -A dp/dx - (lambda a^2 / (2 D A)) phi|phi| / p
with a^2 = Z Rs T the isothermal wave speed squared.
"""

import numpy as np
from gas_properties import mixture_properties, darcy_friction


# --------------------------------------------------------------------------
# Network definition
# --------------------------------------------------------------------------
class Network:
    def __init__(self):
        # 6 nodes: N0 is the supply/compressor (pressure source).
        self.n_nodes = 6
        self.source = 0
        self.p_source_nom = 66.0e5          # 66 bar nominal compressor outlet
        # pipes: (from, to, length[m], diameter[m])
        self.pipes = [
            (0, 1, 60_000, 1.00),   # P0 supply trunk
            (1, 2, 45_000, 0.90),   # P1
            (2, 3, 50_000, 0.90),   # P2
            (3, 4, 55_000, 1.00),   # P3
            (4, 1, 40_000, 0.80),   # P4  -> loop N1-N2-N3-N4-N1
            (3, 5, 48_000, 0.80),   # P5  branch to N5
            (4, 5, 42_000, 0.80),   # P6  -> loop N3-N4-N5
        ]
        self.n_pipes = len(self.pipes)
        # Nominal steady demands [kg/s] withdrawn at nodes (source supplies them)
        self.demand_nom = np.array([0.0, 20.0, 45.0, 55.0, 40.0, 35.0])
        # incidence matrix  B[node, pipe] = +1 if pipe leaves node, -1 if enters
        self.inc = np.zeros((self.n_nodes, self.n_pipes))
        for k, (a, b, L, D) in enumerate(self.pipes):
            self.inc[a, k] = +1.0   # flow defined positive from a -> b (leaves a)
            self.inc[b, k] = -1.0
        self.area = np.array([np.pi * D**2 / 4 for (_, _, _, D) in self.pipes])
        self.length = np.array([L for (_, _, L, _) in self.pipes])
        self.diam = np.array([D for (_, _, _, D) in self.pipes])
        self.total_km = self.length.sum() / 1000.0


# --------------------------------------------------------------------------
# Steady state (nonlinear) - used for operating point + RLC linearisation
# --------------------------------------------------------------------------
def steady_state(net, props, demand=None, p_source=None, x0=None, eps=3.0e-5):
    """
    Solve the nonlinear steady flow: find node pressures and pipe flows s.t.
    mass balance at every node and Weymouth-type pressure drop on every pipe.
    Returns (P[node], Q[pipe]).  Optional x0 = [P_nonsource_bar; Q] warm start.

    The residual is NON-DIMENSIONALISED: the nodal mass balance is divided by a
    characteristic flow and the (squared-pressure) pipe equation by p_src^2.
    Without this the two blocks differ by ~11 orders of magnitude, the trust
    region of the Powell hybrid solver is set by the pipe block alone, and the
    solve fails from a warm start - which in turn makes it impossible to
    continue the operating point into flow reversal.
    """
    from scipy.optimize import root
    if demand is None:
        demand = net.demand_nom
    if p_source is None:
        p_source = net.p_source_nom
    a2 = props['a2']
    nN, nP = net.n_nodes, net.n_pipes
    pS = p_source ** 2                                   # Pa^2 scale
    qS = max(float(np.abs(demand).max()), 1.0)           # kg/s scale

    def residual(x):
        P_bar = np.concatenate([[p_source / 1e5], x[:nN - 1]])   # bar
        Q_ = x[nN - 1:]
        r = np.zeros(nN - 1 + nP)
        # node mass balance (non-source), scaled
        r[:nN - 1] = (net.inc @ Q_ + demand)[1:] / qS
        # pipe eqs (Pa^2), scaled
        for k, (a, b, L, D) in enumerate(net.pipes):
            A = net.area[k]
            lam = darcy_friction(props['rho'], props['mu'], D, Q_[k], eps=eps)
            coef = lam * a2 * L / (D * A**2)     # Pa^2 per (kg/s)^2
            r[nN - 1 + k] = ((P_bar[a] * 1e5)**2 - (P_bar[b] * 1e5)**2
                             - coef * Q_[k] * abs(Q_[k])) / pS
        return r

    if x0 is None:
        x0 = np.concatenate([np.full(nN - 1, p_source / 1e5 - 2.0),
                             np.full(nP, 30.0)])
    sol = root(residual, x0, method='hybr', tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"steady_state did not converge: {sol.message}")
    P_ = np.concatenate([[p_source], sol.x[:nN - 1] * 1e5])   # back to Pa
    Q_ = sol.x[nN - 1:]
    return P_, Q_


def steady_continuation(net, props, demand_target, P0=None, Q0=None, steps=40,
                        p_source=None, eps=3.0e-5):
    """Ramp the demand vector from nominal to target with warm starts.

    Required to reach operating points that reverse one or more loop chords:
    a direct Newton solve from a cold start either misses those branches or
    lands on a different root.
    """
    if P0 is None or Q0 is None:
        P0, Q0 = steady_state(net, props, p_source=p_source, eps=eps)
    x = np.concatenate([P0[1:] / 1e5, Q0])
    P, Q = P0, Q0
    for f in np.linspace(0.0, 1.0, steps)[1:]:
        dem = net.demand_nom + f * (np.asarray(demand_target) - net.demand_nom)
        P, Q = steady_state(net, props, demand=dem, p_source=p_source, x0=x,
                            eps=eps)
        x = np.concatenate([P[1:] / 1e5, Q])
    return P, Q


# --------------------------------------------------------------------------
# Reduced-order RLC lumped state-space (LTI)
# --------------------------------------------------------------------------
def build_rlc(net, props, P0, Q0):
    """
    Build LTI matrices for x=[P_nonsource(nN-1); Q(nP)], driven by
    u=[p_source; demand(nN)].  Continuous:  dx/dt = A x + B u.

    Node:  C_n dP_n/dt = -sum_k inc[n,k] Q_k - demand_n
    Pipe:  L_k dQ_k/dt = (P_a - P_b) - R_k Q_k     (friction linearised at Q0)
    """
    a2 = props['a2']
    nN, nP = net.n_nodes, net.n_pipes

    # node capacitance C_n = V_n / a2, V_n = sum adjacent (A_k L_k /2)
    Vn = np.zeros(nN)
    for k, (a, b, L, D) in enumerate(net.pipes):
        Vn[a] += net.area[k] * net.length[k] / 2.0
        Vn[b] += net.area[k] * net.length[k] / 2.0
    Cn = Vn / a2

    # pipe inductance and linearised resistance
    Lk = net.length / net.area
    Rk = np.zeros(nP)
    for k, (a, b, L, D) in enumerate(net.pipes):
        A = net.area[k]
        lam = darcy_friction(props['rho'], props['mu'], D, Q0[k])
        p_avg = 0.5 * (P0[a] + P0[b])
        # nonlinear drop coefficient in dQ/dt form: (L/A)*(lam a2/(2DA))/p_avg * Q|Q|
        # linearised slope wrt Q at Q0: R_k = (L/A)*(lam a2/(2DA))*2|Q0|/p_avg
        Rk[k] = (L / A) * (lam * a2 / (2 * D * A)) * 2 * abs(Q0[k]) / p_avg
    Lk_eff = net.length / net.area  # = L/A prefactor already folded; keep L dQ/dt form

    # State-space assembly
    ns = (nN - 1) + nP
    A_c = np.zeros((ns, ns))
    B_c = np.zeros((ns, nN + 1))   # inputs: [p_source, demand_0..demand_{nN-1}]

    # map non-source node index -> state index
    node_state = {n: i for i, n in enumerate([n for n in range(nN) if n != net.source])}

    # Node pressure dynamics
    for n in range(nN):
        if n == net.source:
            continue
        si = node_state[n]
        for k in range(nP):
            coef = -net.inc[n, k] / Cn[n]      # dP_n/dt gets -inc*Q/C
            A_c[si, (nN - 1) + k] += coef
        # demand disturbance
        B_c[si, 1 + n] += -1.0 / Cn[n]

    # Pipe flow dynamics: L dQ/dt = P_a - P_b - R Q  -> dQ/dt = (P_a-P_b-R Q)/L
    for k, (a, b, L, D) in enumerate(net.pipes):
        row = (nN - 1) + k
        Lval = Lk_eff[k]
        # P_a term
        if a == net.source:
            B_c[row, 0] += 1.0 / Lval        # p_source input
        else:
            A_c[row, node_state[a]] += 1.0 / Lval
        # -P_b term
        if b == net.source:
            B_c[row, 0] += -1.0 / Lval
        else:
            A_c[row, node_state[b]] += -1.0 / Lval
        # -R Q term
        A_c[row, row] += -Rk[k] / Lval

    # Output: all node pressures (reconstruct source separately) -> measure non-source nodes
    C_out = np.zeros((nN - 1, ns))
    for i in range(nN - 1):
        C_out[i, i] = 1.0

    info = dict(Cn=Cn, Lk=Lk_eff, Rk=Rk, node_state=node_state, ns=ns,
                n_inputs=nN + 1)
    return A_c, B_c, C_out, info


# --------------------------------------------------------------------------
# High-fidelity 1D transient reference (nonlinear, fine grid)
# --------------------------------------------------------------------------
def build_reference(net, props, M=20, Q_basis=None):
    """
    Fine staggered finite-volume discretisation. Each pipe has M interior
    pressure cells and M+1 flow faces. Returns (rhs, unpack, pack, meta).
    State layout: [ node_pressures(nN) , per-pipe: cells(M), faces(M+1) ].
    Source node pressure is overwritten by u(t) each step.

    Q_basis : retained for interface compatibility only. The reference does NOT
              freeze the friction factor: it recomputes a velocity-dependent
              (Swamee-Jain) Darcy factor from the local Reynolds number at every
              step, and keeps the full nonlinear phi|phi|/p momentum term and the
              convective term. This is deliberate - the reference must be richer
              than the reduced model for the comparison to measure reduction
              error rather than grid refinement.
    """
    a2 = props['a2']
    nN, nP = net.n_nodes, net.n_pipes

    # geometry per pipe
    dx = net.length / (M + 1)          # M+1 intervals between M+2 pressure points
    # index bookkeeping
    idx = {}
    ptr = 0
    idx['node'] = slice(ptr, ptr + nN); ptr += nN
    idx['pipe'] = []
    for k in range(nP):
        cell = slice(ptr, ptr + M); ptr += M
        face = slice(ptr, ptr + (M + 1)); ptr += (M + 1)
        idx['pipe'].append((cell, face))
    n_state = ptr

    # node control volume for capacitance (half of adjacent end intervals)
    Vn = np.zeros(nN)
    for k, (a, b, L, D) in enumerate(net.pipes):
        Vn[a] += net.area[k] * dx[k] / 2.0
        Vn[b] += net.area[k] * dx[k] / 2.0
    Cn = Vn / a2

    def rhs(t, y, p_source, demand):
        dydt = np.zeros_like(y)
        P = y[idx['node']].copy()
        P[net.source] = p_source          # enforce source BC
        node_inflow = np.zeros(nN)
        for k, (a, b, L, D) in enumerate(net.pipes):
            A = net.area[k]
            cell_sl, face_sl = idx['pipe'][k]
            pc = y[cell_sl]                # M interior cell pressures
            q = y[face_sl]                 # M+1 faces
            # pressure points along pipe: [P_a, pc_0..pc_{M-1}, P_b]
            pts = np.empty(M + 2)
            pts[0] = P[a]; pts[1:M+1] = pc; pts[M+1] = P[b]
            # face momentum: dq/dt = (A/dx)(pL - pR) - convective - friction
            pL = pts[:-1]; pR = pts[1:]
            pface = 0.5 * (pL + pR)
            # velocity-dependent (Swamee-Jain) turbulent friction, recomputed each
            # step from the local Reynolds number -- richer than the reduced
            # model's fixed-point linearisation.
            rho_f = pface / a2
            vf = np.abs(q) / (rho_f * A + 1e-30)
            Re = rho_f * vf * D / props['mu'] + 1e3
            lam = 0.25 / (np.log10(3.0e-5/(3.7*D) + 5.74/Re**0.9))**2
            fric = lam * a2 / (2 * D * A) * q * np.abs(q) / pface
            # convective momentum term d(phi v)/dx (retained here, dropped in the
            # reduced isothermal model); small in the subsonic regime.
            mom = q * vf
            dmom = np.zeros_like(q)
            dmom[1:-1] = (mom[2:] - mom[:-2]) / (2*dx[k])
            dq = (A / dx[k]) * (pL - pR) - fric - dmom
            dydt[face_sl] = dq
            # cell mass balance: (A dx/a2) dpc/dt = q_in - q_out
            Ccell = A * dx[k] / a2
            dpc = (q[:-1] - q[1:]) / Ccell
            dydt[cell_sl] = dpc
            # node inflow contributions (q[0] enters a as leaving -> a loses q[0]; b gains q[M])
            node_inflow[a] -= q[0]
            node_inflow[b] += q[M]
        # node pressure dynamics (non-source)
        for n in range(nN):
            if n == net.source:
                dydt[idx['node']][n] = 0.0
            else:
                dydt[idx['node'].start + n] = (node_inflow[n] - demand[n]) / Cn[n]
        return dydt

    def init_state(P0, Q0):
        y = np.zeros(n_state)
        y[idx['node']] = P0
        for k, (a, b, L, D) in enumerate(net.pipes):
            cell_sl, face_sl = idx['pipe'][k]
            # linear pressure profile between end nodes
            y[cell_sl] = np.linspace(P0[a], P0[b], M + 2)[1:M+1]
            y[face_sl] = Q0[k]
        return y

    meta = dict(idx=idx, n_state=n_state, M=M, Cn=Cn, dx=dx)
    return rhs, init_state, meta


if __name__ == "__main__":
    net = Network()
    props = mixture_properties(0.0)
    print(f"Network: {net.n_nodes} nodes, {net.n_pipes} pipes, {net.total_km:.0f} km total")
    P0, Q0 = steady_state(net, props)
    print("Steady node pressures [bar]:", np.round(P0 / 1e5, 2))
    print("Steady pipe flows [kg/s]:   ", np.round(Q0, 2))
    print("Mass balance residual [kg/s]:", np.round((net.inc @ Q0 + net.demand_nom)[1:], 4))
    print("Total supply [kg/s]:", round(Q0[0], 2), "= total demand:", net.demand_nom.sum())
