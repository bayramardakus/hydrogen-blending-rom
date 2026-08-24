"""estimator.py - scaled state-space handling and a physically-tuned observer.

Two defects of the previous implementation are fixed here.

SCALING.  The RLC state vector mixes node pressures in Pa (order 1e6-1e7) with
pipe mass flows in kg/s (order 1e1-1e2). The resulting controllability and
observability Gramians have condition numbers around 1e16, so a numerical rank
test reports them as rank-deficient even though the pair is structurally
controllable/observable, and the Riccati solutions are poorly conditioned. Every
system-theoretic computation here is therefore performed on the
NON-DIMENSIONALISED model

    x~ = T x ,   T = diag(1/p_nom, ..., 1/phi_nom, ...)
    A~ = T A T^-1 ,   B~ = T B ,   C~ = C T^-1 .

NOISE MODEL.  The previous tuning used R = 1e-3 * I with the states in Pa, i.e.
it declared a pressure-transducer standard deviation of 0.03 Pa (3e-7 mbar).
The resulting filter gain has ||L||_inf = 1: it passes the raw measurement
through and differentiates it into the unmeasured flow states, so at a realistic
0.5 % transducer noise the flow estimates exceed the true flows by two orders of
magnitude. Here

    R = sigma_v^2 I ,  sigma_v = class * p_nom     (class = 0.1-0.5 % of reading,
                                                    typical of transmission-grade
                                                    pressure transmitters)
    Q = Bd_d Sigma_d Bd_d^T + eps I ,              Sigma_d = demand-forecast
                                                   error covariance,

so both covariances are traceable to a stated physical uncertainty rather than
chosen for numerical convenience.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are


SENSOR_CLASS = 0.0025      # 0.25 % of reading, 1 sigma (mid transmission grade)
DEMAND_ERR = 0.05          # 5 % of nominal withdrawal, 1 sigma


def scaling(P0, Q0, nN):
    """Diagonal non-dimensionalising transform for x = [P_nonsource; phi]."""
    p_ref = float(np.mean(P0[1:]))
    q_ref = max(float(np.mean(np.abs(Q0))), 1.0)
    d = np.concatenate([np.full(nN - 1, 1.0 / p_ref),
                        np.full(len(Q0), 1.0 / q_ref)])
    return np.diag(d), np.diag(1.0 / d), p_ref, q_ref


def discretise(A, B, Ts):
    n, m = A.shape[0], B.shape[1]
    Mb = np.zeros((n + m, n + m))
    Mb[:n, :n] = A
    Mb[:n, n:] = B
    Md = expm(Mb * Ts)
    return Md[:n, :n], Md[:n, n:]


def build_observer(A_c, B_c, C_out, P0, Q0, net, Ts,
                   sensor_class=SENSOR_CLASS, demand_err=DEMAND_ERR,
                   meas_nodes=None):
    """Steady-state Kalman filter on the non-dimensionalised model.

    Returns a dict with the scaled discrete model, the gain L, the physical
    measurement standard deviation, and the transforms.
    """
    nN = net.n_nodes
    T, Ti, p_ref, q_ref = scaling(P0, Q0, nN)
    A = T @ A_c @ Ti
    # The control input is the supply pressure. It is expressed in the SAME
    # normalised pressure unit as the states, so its input matrix must be
    # scaled by p_ref as well - otherwise the actuator appears ~1e6 times
    # weaker than it is and the controller stops using its authority.
    Bu = T @ B_c[:, [0]] * p_ref
    # The demand disturbance is normalised by a characteristic flow for the
    # same reason, so that all three input/state blocks are O(1) and the
    # predictive QP is well conditioned.
    Bd = T @ B_c[:, 1:] * q_ref
    # C_out is a 0/1 selector on the node-pressure block. Because those states
    # are already normalised by p_ref, the SAME selector maps the scaled state
    # to a scaled (dimensionless) output - no further transformation, and the
    # measurement fed to the filter must likewise be divided by p_ref.
    C = C_out
    Ad, Bud = discretise(A, Bu, Ts)
    _, Bdd = discretise(A, Bd, Ts)

    # measurement noise: sensor_class of the local reading, expressed in the
    # scaled output (outputs are pressures, scaled by 1/p_ref)
    # meas_nodes lets a SUBSET of the node pressures be instrumented; C_out must
    # then carry exactly those rows. Default: every non-supply node.
    if meas_nodes is None:
        meas_nodes = [n for n in range(nN) if n != net.source]
    sig_v_pa = sensor_class * P0[np.asarray(meas_nodes, dtype=int)]
    if C.shape[0] != len(sig_v_pa):
        raise ValueError("C_out row count must match meas_nodes")
    Rk = np.diag((sig_v_pa / p_ref) ** 2)

    # process noise: the dominant unmodelled input is the demand forecast error
    sig_d = demand_err * np.maximum(net.demand_nom, 1.0) / q_ref
    Qk = Bdd @ np.diag(sig_d ** 2) @ Bdd.T
    Qk += 1e-12 * np.eye(A.shape[0])          # regularisation, keeps Q >= 0

    Pk = solve_discrete_are(Ad.T, C.T, Qk, Rk)
    L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rk)
    return dict(Ad=Ad, Bud=Bud, Bdd=Bdd, Cd=C, L=L, Pk=Pk, Qk=Qk, Rk=Rk,
                T=T, Ti=Ti, p_ref=p_ref, q_ref=q_ref, sig_v_pa=sig_v_pa,
                gain_inf=float(np.abs(L).max()))


# ----------------------------------------------------------------------
def pbh_controllable(A, B, tol=1e-9):
    """Popov-Belevitch-Hautus controllability test.

    Preferred over a Gramian rank test: it does not require solving a Lyapunov
    equation whose condition number is the square of the model's, and it
    reports WHICH mode is deficient.
    """
    n = A.shape[0]
    lams = np.linalg.eigvals(A)
    worst, worst_lam = np.inf, None
    for lam in lams:
        Mx = np.hstack([A - lam * np.eye(n), B])
        s = np.linalg.svd(Mx, compute_uv=False)
        if s[-1] < worst:
            worst, worst_lam = s[-1], lam
    scale = np.linalg.svd(np.hstack([A, B]), compute_uv=False)[0]
    return dict(ok=bool(worst / scale > tol), min_sv=float(worst),
                rel=float(worst / scale), mode=complex(worst_lam))


def pbh_observable(A, C, tol=1e-9):
    r = pbh_controllable(A.T, C.T, tol)
    return r


def hankel_spectrum(A, B, C):
    """Hankel singular values of the (stable) system - a scaling-robust
    replacement for a Gramian rank test."""
    from scipy.linalg import solve_continuous_lyapunov, cholesky
    Wc = solve_continuous_lyapunov(A, -B @ B.T)
    Wo = solve_continuous_lyapunov(A.T, -C.T @ C)
    Wc = (Wc + Wc.T) / 2
    Wo = (Wo + Wo.T) / 2
    ev = np.linalg.eigvals(Wc @ Wo)
    hsv = np.sqrt(np.abs(np.real(ev)))
    return np.sort(hsv)[::-1]
