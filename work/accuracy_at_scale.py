"""
accuracy_at_scale.py
--------------------
Show that the sub-percent RLC accuracy is not an artefact of the 6-node
didactic network: repeat the reduced-vs-reference validation on a larger
(~35-node) meshed network with SciGRID-like parameters.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from gas_properties import mixture_properties
import network as NW
from scalability import GenNet


def find_convergent_net(N, props, seeds=range(40)):
    for sd in seeds:
        net = GenNet(N, seed=sd)
        net.demand_nom = np.clip(net.demand_nom * 0.25, 2.0, 12.0)  # modest loads
        net.demand_nom[net.source] = 0.0
        try:
            P0, Q0 = NW.steady_state(net, props)
            if np.all(P0 > 40e5) and np.all(np.isfinite(Q0)):
                return net, P0, Q0, sd
        except Exception:
            continue
    return None


def run(N=35, x_h2=0.05, M_ref=12):
    props = mixture_properties(x_h2)
    got = find_convergent_net(N, props)
    if got is None:
        print("no convergent network found"); return
    net, P0, Q0, sd = got
    nN, nP = net.n_nodes, net.n_pipes
    print(f"Larger network: {nN} nodes, {nP} pipes, {net.total_km:.0f} km (seed {sd})")

    A_c, B_c, C_out, info = NW.build_rlc(net, props, P0, Q0)
    d0 = net.demand_nom.copy()

    # disturbance: surge at two random delivery nodes
    rng = np.random.default_rng(1)
    n1, n2 = rng.choice(range(1, nN), 2, replace=False)
    def demand(t):
        d = d0.copy()
        d[n1] += 8.0*np.clip((t-1800)/300, 0, 1)
        d[n2] += 6.0*np.clip((t-3600)/300, 0, 1)
        return d

    t_eval = np.linspace(0, 7200, 241)
    # reduced (deviation form)
    def rlc_rhs(t, dx):
        du = np.concatenate([[0.0], demand(t)-d0])
        return A_c @ dx + B_c @ du
    t0 = time.perf_counter()
    sol_r = solve_ivp(rlc_rhs, [0,7200], np.zeros(info['ns']), t_eval=t_eval,
                      method="RK45", rtol=1e-6, atol=1e-6)
    t_rlc = time.perf_counter()-t0
    P_rlc = P0[1:,None] + sol_r.y[:nN-1,:]

    # reference
    rhs, init_state, meta = NW.build_reference(net, props, M=M_ref, Q_basis=Q0)
    y0 = init_state(P0, Q0)
    def ref_rhs(t, y): return rhs(t, y, P0[0], demand(t))
    t0 = time.perf_counter()
    sol_f = solve_ivp(ref_rhs, [0,7200], y0, t_eval=t_eval, method="LSODA",
                      rtol=1e-6, atol=1e-3)
    t_ref = time.perf_counter()-t0
    idxN = meta['idx']['node']
    P_ref = sol_f.y[idxN,:][1:,:]

    err = P_rlc/1e5 - P_ref/1e5
    mape = np.mean(np.abs(err)/np.abs(P_ref/1e5))*100
    maxe = np.max(np.abs(err))
    print(f"  reference states = {meta['n_state']}, RLC states = {info['ns']}")
    print(f"  pressure MAPE = {mape:.3f}%   max error = {maxe:.3f} bar")
    print(f"  CPU: reference = {t_ref*1000:.0f} ms, RLC = {t_rlc*1000:.1f} ms, "
          f"speed-up = {t_ref/t_rlc:.0f}x")
    return dict(nN=nN, nP=nP, km=net.total_km, mape=mape, maxe=maxe,
                n_ref=meta['n_state'], n_rlc=info['ns'], speedup=t_ref/t_rlc)


if __name__ == "__main__":
    for x in [0.05, 0.25]:
        print(f"\n=== H2 = {x*100:.0f} vol% ===")
        run(N=35, x_h2=x)
