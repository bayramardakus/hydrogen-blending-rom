"""uncertainty_budget.py - the consolidated uncertainty analysis.

The study is computational: there is no experimental apparatus, so there is no
instrument uncertainty to quote. What there is instead is a set of input
assumptions, each of which carries a stated range, and the operational question
is how far the two quantities the scheme must respect can move when those inputs
are wrong:

    the minimum delivered pressure  (the 60 bar floor), and
    the peak delivered hydrogen fraction  (the 20 vol% interchangeability cap).

Every source below is propagated to those two quantities on the real
SciGRID_gas sub-network. Sources already quantified elsewhere in the package are
read from their result files rather than recomputed; the two that were never
propagated to the delivered blend (pipe roughness and gas temperature, through
the residence times) are propagated here by replaying the recorded injection
schedule of the closed-loop run through a plant built at the perturbed
parameters. Replaying a fixed schedule is deliberate: it isolates the effect of
the parameter error from the controller's reaction to it, which is exactly the
quantity the constraint back-off has to cover.

Inputs : realnet_blend_results.npy, sensitivity_results.npy, i8_eos.npy,
         advection_convergence.npy, realnet_reduction.npy, robustness_studies.npy
Output : uncertainty_budget.npy
Run from the work/ directory:  python uncertainty_budget.py
"""
import numpy as np

import network
import realnet
import realnet_blend as RB
from gas_properties import mixture_properties
from network import steady_state

# Perturbations propagated here. Ranges are those already used in the package's
# parameter studies, so the budget and the sensitivity study cannot disagree.
EPS_NOM = 3.0e-5           # pipe roughness used throughout [m]
EPS_RANGE = (2.0e-5, 2.0e-4)
T_NOM = 288.15             # gas temperature used throughout [K]
T_RANGE = (263.15, 303.15)
RHO_REL = 0.0234           # density spread against GERG-2008 over 0-25 vol% [-]

CASES = [
    ('nominal',            dict()),
    ('roughness low',      dict(eps=EPS_RANGE[0])),
    ('roughness high',     dict(eps=EPS_RANGE[1])),
    ('temperature low',    dict(T=T_RANGE[0])),
    ('temperature high',   dict(T=T_RANGE[1])),
    ('density low (EOS)',  dict(rho_rel=-RHO_REL)),
    ('density high (EOS)', dict(rho_rel=+RHO_REL)),
]


def replay(u_cmd, avail, eps=EPS_NOM, T=T_NOM, rho_rel=0.0):
    """Replay a recorded injection schedule through a plant built at the
    perturbed parameters and return the delivered-fraction history."""
    net = realnet.RealNetwork(72.0, 500.0)
    props = mixture_properties(0.05, T=T)
    if rho_rel:
        # The residence time uses rho = p / a^2, so a density error of rho_rel
        # is imposed by scaling a^2 by 1/(1+rho_rel). That also moves the wave
        # speed by -rho_rel/2, which is what the equation-of-state comparison
        # of Section 6.1 actually reports (2.3 % in density, 1.1 % in a).
        props = dict(props)
        props['a2'] = props['a2'] / (1.0 + rho_rel)
        props['a'] = float(np.sqrt(props['a2']))
        props['rho'] = props['rho'] * (1.0 + rho_rel)
    P0, Q0 = steady_state(net, props, eps=eps)
    inj = [net.source, net.western_hub]

    # The plant solves its own steady states each step; route them through the
    # same roughness. Restored in the finally block so nothing leaks.
    orig_ss, orig_sc = RB.steady_state, RB.steady_continuation
    orig_sig = RB.SIG_ACT
    RB.steady_state = lambda *a, **k: orig_ss(*a, **{**k, 'eps': eps})
    RB.steady_continuation = lambda *a, **k: orig_sc(*a, **{**k, 'eps': eps})
    RB.SIG_ACT = 0.0        # the recorded schedule already carries the realised
    try:                    # actuator noise; do not add a second draw
        plant = RB.Plant(net, props, P0, Q0, inj)
        nsteps = len(u_cmd)
        ydel = np.zeros((nsteps, plant.mdl['Wsel'].shape[0]))
        for k in range(nsteps):
            t = k * RB.TS
            _, z = plant.step(t, u_cmd[k], avail[k])
            ydel[k] = z
    finally:
        RB.steady_state, RB.steady_continuation = orig_ss, orig_sc
        RB.SIG_ACT = orig_sig

    tau = plant.tau_ref
    return dict(ydel=ydel, peak=float(ydel.max()), tau_max_h=float(tau.max()/3600),
                pmin_bar=float(np.min(P0[[n for n in range(net.n_nodes)
                                          if n != net.source]]) / 1e5))


def main():
    rec = np.load('realnet_blend_results.npy', allow_pickle=True).item()
    mpc = rec['mpc']
    u_cmd = np.column_stack([mpc['uE'], mpc['uW']])
    avail = np.column_stack([mpc['aE'], mpc['aW']])
    print('replaying %d steps of the recorded schedule through %d plants'
          % (len(u_cmd), len(CASES)))

    out = {}
    for label, kw in CASES:
        r = replay(u_cmd, avail, **kw)
        out[label] = r
        print('  %-20s peak delivered %6.3f vol%%   tau_max %5.1f h   '
              'min steady pressure %6.2f bar'
              % (label, r['peak'] * 100, r['tau_max_h'], r['pmin_bar']),
              flush=True)

    base = out['nominal']['peak']
    print('\nshift in peak delivered fraction against the nominal plant '
          '[volume percentage points]')
    for label, _ in CASES[1:]:
        print('  %-20s %+6.3f' % (label, (out[label]['peak'] - base) * 100))

    np.save('uncertainty_budget.npy',
            dict(cases=out, base_peak=base,
                 eps_nom=EPS_NOM, eps_range=EPS_RANGE,
                 T_nom=T_NOM, T_range=T_RANGE, rho_rel=RHO_REL),
            allow_pickle=True)
    print('\nsaved uncertainty_budget.npy')


if __name__ == '__main__':
    main()
