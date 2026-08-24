"""transport_fir.py - accurate, control-ready hydrogen-transport representations.

Motivation
----------
The multi-cell upwind model of `composition_adv.py` is a tanks-in-series
approximation of plug flow: the residence-time distribution of a pipe
discretised into M cells has relative standard deviation

        sigma_tau / tau = 1 / sqrt(M) ,

so the model carries a numerical diffusion that only vanishes as M -> infinity.
At M = 4 the front is smeared by +/- 50 % of the residence time, which
under-predicts the PEAK delivered fraction of a transient injection by ~9
volume-percentage points at far nodes - the same safety-relevant failure mode
for which the single-cell (M = 1) model was rejected.

Raising M inside the predictive controller is expensive: the state count, and
hence the size of every optimisation, grows linearly with M.

Resolution
----------
The transport model is linear and time-invariant at a frozen flow field, so the
map from the injection sequence to the delivered composition at every node is a
CONVOLUTION.  Its Markov parameters (discrete impulse response)

        G_j = W A_d^j B_d ,        j = 0, 1, ..., N-1

can be computed OFFLINE from an arbitrarily fine transport model, and the
predictive controller then works directly with

        z(k) = z_base + sum_j G_j * ( u(k-1-j) - u_base ) .

This decouples accuracy from online cost completely:
  * the predictor is as accurate as the offline model used to build G;
  * the optimisation carries NO transport states - only the H injection
    decisions - so the problem is smaller than the M = 4 state-space version
    while being far more accurate.

Two builders are provided:
  build_cfl_composition : per-pipe CFL-matched upwind (M_k = round(tau_k/Ts)),
                          i.e. Courant number 1, useful as a cheap state-space
                          alternative and as a simulation plant;
  markov_params         : the FIR / impulse-response representation used by the
                          controller.
"""
import numpy as np
from scipy.linalg import expm

from composition import build_composition_model


# ----------------------------------------------------------------------
def build_upwind(net, props, P0, Q0, Mk, inj_nodes=None):
    """Multi-cell upwind transport with a PER-PIPE cell count Mk[k].

    Generalises `composition_adv.build_adv_composition` (which uses a single M
    for every pipe) so that the discretisation can be matched to each pipe's
    residence time.

    Returns dict(A, B, Wsel, tau, Mk, nstate, offsets, inj_nodes).
    """
    cm = build_composition_model(net, props, P0, Q0)
    tau, up, Wn = cm['tau'], cm['upstream'], cm['Wnode']
    nP, nN, src = net.n_pipes, net.n_nodes, net.source
    if inj_nodes is None:
        inj_nodes = [src]
    inj_index = {n: j for j, n in enumerate(inj_nodes)}

    Mk = np.maximum(1, np.asarray(Mk, dtype=int))
    if Mk.size == 1:
        Mk = np.full(nP, int(Mk))
    off = np.concatenate([[0], np.cumsum(Mk)]).astype(int)
    ns = int(off[-1])

    A = np.zeros((ns, ns))
    B = np.zeros((ns, len(inj_nodes)))
    out = lambda k: off[k] + Mk[k] - 1

    for k in range(nP):
        r = Mk[k] / tau[k]
        for i in range(Mk[k]):
            idx = off[k] + i
            A[idx, idx] -= r
            if i > 0:
                A[idx, off[k] + i - 1] += r
            else:
                u = up[k]
                if u in inj_index:
                    B[idx, inj_index[u]] += r
                else:
                    for j in range(nP):
                        if Wn[u, j]:
                            A[idx, out(j)] += r * Wn[u, j]

    Wsel = np.zeros((nN, ns))
    for n in range(nN):
        for k in range(nP):
            if Wn[n, k]:
                Wsel[n, out(k)] += Wn[n, k]

    return dict(A=A, B=B, Wsel=Wsel, tau=tau, Mk=Mk, nstate=ns, offsets=off,
                inj_nodes=list(inj_nodes), Wnode=Wn, upstream=up)


def build_cfl_composition(net, props, P0, Q0, Ts, inj_nodes=None, refine=1):
    """CFL-matched upwind: M_k = max(1, round(refine * tau_k / Ts)).

    With Courant number sigma_k = M_k Ts / tau_k = 1 the explicit upwind update
    is the exact shift operator, so at the sampling instants the scheme carries
    no numerical diffusion for that pipe.  `refine` > 1 sub-samples further.
    """
    cm = build_composition_model(net, props, P0, Q0)
    Mk = np.maximum(1, np.round(refine * cm['tau'] / Ts).astype(int))
    return build_upwind(net, props, P0, Q0, Mk, inj_nodes)


# ----------------------------------------------------------------------
def discretise(A, B, Ts):
    """Exact zero-order-hold discretisation."""
    n, m = A.shape[0], B.shape[1]
    Mb = np.zeros((n + m, n + m))
    Mb[:n, :n] = A
    Mb[:n, n:] = B
    Md = expm(Mb * Ts)
    return Md[:n, :n], Md[:n, n:]


def markov_params(A, B, W, Ts, N, rows=None):
    """Discrete impulse response  G_j = W A_d^j B_d  for j = 0 .. N-1.

    Parameters
    ----------
    A, B, W : continuous-time transport model and node-selection matrix
    Ts      : controller sampling interval [s]
    N       : number of taps (must cover the slowest delivery delay:
              N >= 3 * tau_max / Ts is a safe rule of thumb)
    rows    : optional list of node indices to retain (default: all)

    Returns
    -------
    G : ndarray, shape (N, n_rows, n_inputs)
    """
    Ad, Bd = discretise(A, B, Ts)
    Wr = W if rows is None else W[rows, :]
    G = np.empty((N, Wr.shape[0], Bd.shape[1]))
    V = Bd.copy()                      # A_d^j B_d
    for j in range(N):
        G[j] = Wr @ V
        V = Ad @ V
    return G


def fir_settling(G, tol=1e-3):
    """Index after which every impulse-response tap is below `tol` of the
    steady gain - i.e. the number of taps actually needed."""
    S = np.cumsum(np.abs(G).sum(axis=(1, 2)))
    S = S / S[-1]
    return int(np.searchsorted(S, 1.0 - tol)) + 1


def fir_predict(G, u_hist, u_future, u_base):
    """Delivered composition over the horizon from an FIR model.

    z(k) = z_base + sum_j G_j ( u(k-1-j) - u_base )

    u_hist   : (N, n_inp) past inputs, u_hist[0] = most recent (t-1)
    u_future : (H, n_inp) planned inputs, u_future[0] applied at t
    Returns z of shape (H, n_rows), already including the base offset.
    """
    N, nr, ni = G.shape
    H = u_future.shape[0]
    z = np.full((H, nr), 0.0)
    for i in range(H):
        for j in range(N):
            idx = i - 1 - j                     # index into u_future
            if idx >= 0:
                du = u_future[idx] - u_base
            else:
                h = -idx - 1                    # index into u_hist
                if h >= u_hist.shape[0]:
                    break
                du = u_hist[h] - u_base
            z[i] += G[j] @ du
    return z + u_base


def fir_matrices(G, H):
    """Block-Toeplitz matrices for the predictive controller.

    Returns (Phi, Psi) such that, with du_future (H, n_inp) and du_hist
    (N, n_inp) both measured from u_base,

        z_dev = Phi @ vec(du_future) + Psi @ vec(du_hist)

    where z_dev has shape (H * n_rows,).  Phi is the forced-response
    (controllable) block used inside the optimisation; Psi @ vec(du_hist) is the
    free response of the already-committed injection history and enters the
    problem as a constant.
    """
    N, nr, ni = G.shape
    Phi = np.zeros((H * nr, H * ni))
    Psi = np.zeros((H * nr, N * ni))
    for i in range(H):
        for j in range(N):
            idx = i - 1 - j
            if idx >= 0:
                Phi[i*nr:(i+1)*nr, idx*ni:(idx+1)*ni] += G[j]
            else:
                h = -idx - 1
                if h < N:
                    Psi[i*nr:(i+1)*nr, h*ni:(h+1)*ni] += G[j]
    return Phi, Psi


if __name__ == "__main__":
    from gas_properties import mixture_properties
    from network import Network, steady_state
    from composition_adv import build_adv_composition

    net = Network()
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)
    cm = build_composition_model(net, props, P0, Q0)
    Ts = 1800.0
    print("residence times tau [h]:", np.round(cm['tau'] / 3600, 1))

    fine = build_upwind(net, props, P0, Q0, np.full(net.n_pipes, 60))
    G = markov_params(fine['A'], fine['B'], fine['Wsel'], Ts, N=300,
                      rows=[1, 2, 3, 4, 5])
    print(f"fine model states      : {fine['nstate']}")
    print(f"FIR taps needed (99.9%): {fir_settling(G)} of 300 "
          f"({fir_settling(G)*Ts/3600:.1f} h)")
    print(f"steady gain per node   : "
          f"{np.round(G.sum(axis=0)[:, 0], 4)}   (should all be 1.0)")

    cfl = build_cfl_composition(net, props, P0, Q0, Ts)
    print(f"CFL-matched M_k        : {cfl['Mk'].tolist()}  "
          f"-> {cfl['nstate']} states")
    m4 = build_adv_composition(net, props, P0, Q0, M=4)
    print(f"uniform M=4            : {m4['nstate']} states")
    print("\nMPC decision variables: FIR -> H per injector (no transport "
          "states); state-space -> H*(states+inputs).")
