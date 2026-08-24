"""
composition.py
--------------
Hydrogen composition-transport model for the network. The hydrogen fraction
advects with the gas: each pipe carries the blend from its upstream node to its
downstream node with a residence-time lag, and each junction mixes the incoming
streams in proportion to their (nominal) mass flows. This makes the *delivered
hydrogen fraction* a dynamic, spatially-distributed variable driven by the
injection at the source(s) -- the genuinely hydrogen-specific part of the model.

States: pipe-outlet hydrogen fractions y_k (one per pipe).
Node fractions are algebraic, flow-weighted mixes of incoming pipe outlets
(valid because the nominal-flow directions form a DAG).
Input: injection hydrogen fraction y_inj at the source node.

Because gas velocities in transmission pipe are only a few m/s, residence times
are on the order of hours -- so an injection change reaches distant customers
with a large, network-dependent delay. This slow composition transport, coupled
to the fast (seconds) pressure dynamics, is what the predictive controller must
anticipate.
"""
import numpy as np
from gas_properties import mixture_properties
from network import Network, steady_state


def build_composition_model(net, props, P0, Q0, y_nom=0.05):
    """
    Returns dict with linear composition dynamics  dy/dt = Ay y + By u_inj,
    node-fraction reconstruction  y_node = Ny y (+ Ninj u_inj), residence times,
    and the flow-direction bookkeeping.
    """
    nN, nP = net.n_nodes, net.n_pipes
    # flow direction per pipe from steady solution
    upstream = np.zeros(nP, dtype=int)
    downstream = np.zeros(nP, dtype=int)
    for k, (a, b, L, D) in enumerate(net.pipes):
        if Q0[k] >= 0:
            upstream[k], downstream[k] = a, b
        else:
            upstream[k], downstream[k] = b, a

    # incoming pipes per node (pipes whose downstream == node)
    incoming = {n: [k for k in range(nP) if downstream[k] == n] for n in range(nN)}

    # residence time per pipe  tau = mass_in_pipe / mass_flow = rho*A*L/|phi|
    a2 = props['a2']
    tau = np.zeros(nP)
    for k, (a, b, L, D) in enumerate(net.pipes):
        p_avg = 0.5 * (P0[a] + P0[b])
        rho_k = p_avg / a2            # rho = p/(Z Rs T) = p/a2
        tau[k] = rho_k * net.area[k] * net.length[k] / max(abs(Q0[k]), 1e-6)

    # node fraction weights (flow-weighted mix of incoming pipe outlets)
    # y_node(n) = sum_{k in incoming(n)} w[n,k] * y_k     (algebraic)
    Wnode = np.zeros((nN, nP))
    for n in range(nN):
        ks = incoming[n]
        if not ks:
            continue
        tot = sum(abs(Q0[k]) for k in ks)
        for k in ks:
            Wnode[n, k] = abs(Q0[k]) / tot

    # pipe dynamics: dy_k/dt = (1/tau_k)(y_up(k) - y_k)
    # y_up(k) = y_inj if upstream==source else y_node(upstream) = Wnode[up] . y
    Ay = np.zeros((nP, nP))
    By = np.zeros((nP, 1))
    for k in range(nP):
        u = upstream[k]
        Ay[k, k] += -1.0 / tau[k]
        if u == net.source:
            By[k, 0] += 1.0 / tau[k]
        else:
            Ay[k, :] += (1.0 / tau[k]) * Wnode[u, :]

    info = dict(Ay=Ay, By=By, Wnode=Wnode, tau=tau, upstream=upstream,
                downstream=downstream, incoming=incoming, y_nom=y_nom)
    return info


def node_fraction(Wnode, y, n, y_inj, source):
    if n == source:
        return y_inj
    w = Wnode[n]
    if w.sum() == 0:
        return y_inj
    return float(w @ y)


if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    net = Network()
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)
    cm = build_composition_model(net, props, P0, Q0)
    print("Flow directions (upstream->downstream):")
    for k in range(net.n_pipes):
        print(f"  P{k}: N{cm['upstream'][k]} -> N{cm['downstream'][k]}  "
              f"tau={cm['tau'][k]/3600:.2f} h  (|Q|={abs(Q0[k]):.1f} kg/s)")
    # step response: injection 5% -> 20% at t=0, watch delivery nodes
    Ay, By = cm['Ay'], cm['By']
    def rhs(t, y): return (Ay @ y + (By[:,0])*0.20)
    y0 = np.full(net.n_pipes, 0.05)
    sol = solve_ivp(rhs, [0, 12*3600], y0, t_eval=np.linspace(0,12*3600,50), method="RK45")
    print("\nDelivery-node H2 fraction after step 5%->20% injection:")
    for n in [2,3,4,5]:
        yn = np.array([node_fraction(cm['Wnode'], sol.y[:,i], n, 0.20, net.source)
                       for i in range(sol.y.shape[1])])
        # time to reach 90% of final (0.20)
        idx = np.argmax(yn >= 0.05 + 0.9*0.15)
        print(f"  N{n}: reaches 18.5% at t={sol.t[idx]/3600:.2f} h  (final {yn[-1]*100:.1f}%)")
