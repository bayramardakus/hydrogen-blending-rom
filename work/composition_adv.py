"""Advection (multi-cell upwind) composition-transport model — plug-flow faithful.
Backward-compatible superset of the first-order model (M=1 reproduces it)."""
import numpy as np
from composition import build_composition_model

def build_adv_composition(net, props, P0, Q0, M=12, inj_nodes=None, tau_max_h=None):
    cm = build_composition_model(net, props, P0, Q0)
    tau = cm['tau'].copy(); up = cm['upstream']; Wnode = cm['Wnode']
    nP, nN, src = net.n_pipes, net.n_nodes, net.source
    if inj_nodes is None: inj_nodes = [src]
    inj_index = {n:j for j,n in enumerate(inj_nodes)}
    if tau_max_h is not None: tau = np.minimum(tau, tau_max_h*3600.0)
    ns = nP*M
    A = np.zeros((ns, ns)); B = np.zeros((ns, len(inj_nodes)))
    o = lambda k: k*M+(M-1)
    for k in range(nP):
        r = M/tau[k]
        for i in range(M):
            idx = k*M+i; A[idx, idx] -= r
            if i > 0: A[idx, k*M+i-1] += r
            else:
                u = up[k]
                if u in inj_index: B[idx, inj_index[u]] += r
                else:
                    for j in range(nP):
                        if Wnode[u, j]: A[idx, o(j)] += r*Wnode[u, j]
    # node -> state selector (outlet cells, flow-weighted)
    Wsel = np.zeros((nN, ns))
    for n in range(nN):
        for k in range(nP):
            if Wnode[n, k]: Wsel[n, o(k)] += Wnode[n, k]
    return dict(A=A, B=B, Wsel=Wsel, tau=tau, o=o, inj_nodes=list(inj_nodes),
                nstate=ns, M=M, Wnode=Wnode)
