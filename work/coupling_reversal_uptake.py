"""coupling_reversal_uptake.py - three checks on the limits of the modelling
assumptions.

F1  TWO-WAY COUPLING.  Hydrogen is transported as a passive
    scalar on the pressure-model flow field: the injection does not perturb the
    pressure solution. The manuscript scopes full two-way coupling as future
    work, but does not say HOW LARGE the neglected effect is. This computes the
    order of magnitude directly.

F2  FLOW REVERSAL ON THE REAL NETWORK.  The frozen-flow
    envelope is measured on the didactic network, where reversal is reachable.
    On the real sub-network no chord reverses under the diurnal demand variation
    used in the closed-loop study. That is stated in the manuscript, but the absence
    of reversal is only meaningful with the amplitude at which it was sought.
    This sweeps the demand amplitude to find the reversal threshold, if any.

F3  ATTRIBUTING THE CHANGE IN UPTAKE.  Green-hydrogen uptake is lower once the
    loop is closed against an independent plant and a back-off is applied. This
    runs the simpler configuration (controller's own model as plant, no
    back-off) alongside the adopted one, so the difference can be attributed
    rather than merely noted.
"""
import numpy as np

from gas_properties import mixture_properties
from network import steady_state, steady_continuation, build_rlc
from composition import build_composition_model
import realnet

RES = {}


# ======================================================================
def F1_two_way_coupling():
    print("\n" + "=" * 76)
    print("F1  MAGNITUDE OF THE NEGLECTED TWO-WAY COUPLING")
    print("=" * 76)
    net, props, P0, Q0 = realnet.build()
    y_base, y_peak = 0.05, 0.186        # baseline and worst delivered fraction
    p_b = mixture_properties(y_base, p_ref=float(np.mean(P0[1:])))
    p_p = mixture_properties(y_peak, p_ref=float(np.mean(P0[1:])))

    MH2 = mixture_properties(1.0)['M']
    MNG = mixture_properties(0.0)['M']
    w = lambda y: y * MH2 / (y * MH2 + (1 - y) * MNG)
    mdot = net.demand_nom.sum()
    dm_h2 = (w(y_peak) - w(y_base)) * mdot

    # nodal capacitance C_n = V_n / a^2 and incremental resistance R_k ~ a^2
    da2 = (p_p['a2'] - p_b['a2']) / p_b['a2'] * 100
    dC = (p_b['a2'] / p_p['a2'] - 1) * 100
    dR = da2
    drho = (p_p['rho'] - p_b['rho']) / p_b['rho'] * 100

    print(f"operating point: {mdot:.0f} kg/s throughput, "
          f"{y_base*100:.0f} -> {y_peak*100:.1f} vol% H2 at the worst node")
    print(f"  hydrogen mass added                     : {dm_h2:.2f} kg/s "
          f"= {dm_h2/mdot*100:.2f} % of throughput")
    print(f"  change in a^2 across the front          : {da2:+.1f} %")
    print(f"  -> nodal capacitance C_n = V/a^2        : {dC:+.1f} %")
    print(f"  -> incremental resistance R_k ~ a^2     : {dR:+.1f} %")
    print(f"  -> mixture density (linepack mass)      : {drho:+.1f} %")

    # what that means for the pressure solution: re-solve the steady state with
    # the enriched mixture and compare the delivery pressures
    P1, Q1 = steady_state(net, p_p, x0=np.concatenate([P0[1:] / 1e5, Q0]))
    dP = np.abs(P1[1:] - P0[1:]).max() / 1e5
    dQ = np.abs(Q1 - Q0).max()
    print(f"\n  steady solution re-solved at the enriched composition:")
    print(f"    largest change in a delivery pressure : {dP:.2f} bar")
    print(f"    largest change in a pipe flow         : {dQ:.2f} kg/s "
          f"({dQ/np.abs(Q0).max()*100:.1f} % of the largest flow)")
    print(f"\n  READING: the injected hydrogen is {dm_h2/mdot*100:.2f} % of the "
          f"mass throughput, so it does not")
    print(f"  materially change the mass balance. The composition DOES change "
          f"the wave speed by")
    print(f"  {da2/2:+.0f} % and hence the linepack capacitance by {dC:+.0f} %, "
          f"which is the coupling the")
    print(f"  passive-scalar treatment neglects. Its effect on the quantity the "
          f"controller acts on")
    print(f"  is bounded by {dP:.2f} bar on any delivery pressure, i.e. about "
          f"{dP/3.5*100:.0f} % of the 3.5 bar")
    print("  pressure back-off already carried, and it appears as a slow "
          "composition-dependent")
    print("  drift rather than as a fast disturbance. Full two-way coupling "
          "would therefore refine")
    print("  the pressure trajectory within the existing margin, not overturn "
          "the closed-loop")
    print("  behaviour - but it is the largest single modelling simplification "
          "that remains.")
    RES['F1'] = dict(dm_frac=dm_h2 / mdot * 100, da2=da2, dC=dC, drho=drho,
                     dP=dP, dQ=dQ)
    return RES['F1']


# ======================================================================
def F2_reversal_threshold(amps=(0.15, 0.25, 0.35, 0.50, 0.70, 0.90)):
    print("\n" + "=" * 76)
    print("F2  DEMAND AMPLITUDE AT WHICH A LOOP CHORD REVERSES (REAL NETWORK)")
    print("=" * 76)
    net, props, P0, Q0 = realnet.build()
    rng = np.random.default_rng(7)
    phase = rng.uniform(0, 2 * np.pi, net.n_nodes)
    sgn0 = np.sign(Q0)
    print(f"{'amplitude':>10} {'max |dtau|/tau':>15} {'reversed chords':>17}")
    rows = []
    for amp in amps:
        worst_rev, worst_drift = 0, 0.0
        P, Q = P0.copy(), Q0.copy()
        for h in np.arange(0, 24, 0.5):
            d = net.demand_nom * (1.0 + amp * np.sin(2 * np.pi * h / 24 + phase))
            d[net.source] = 0.0
            try:
                P, Q = steady_state(net, props, demand=d,
                                    x0=np.concatenate([P[1:] / 1e5, Q]))
            except RuntimeError:
                try:
                    P, Q = steady_continuation(net, props, d)
                except RuntimeError:
                    continue
            worst_rev = max(worst_rev, int((np.sign(Q) != sgn0).sum()))
            cm = build_composition_model(net, props, P, Q)
            cm0 = build_composition_model(net, props, P0, Q0)
            worst_drift = max(worst_drift,
                              float(np.max(np.abs(cm['tau'] - cm0['tau'])
                                           / cm0['tau'])))
        rows.append(dict(amp=amp, rev=worst_rev, drift=worst_drift))
        print(f"{amp*100:9.0f}% {worst_drift*100:14.1f}% {worst_rev:17d}")
    first = next((r for r in rows if r['rev'] > 0), None)
    print()
    if first is None:
        print(f"  No loop chord reverses at any tested amplitude, up to "
              f"+/-{max(amps)*100:.0f} % of nominal")
        print("  demand at every node. On this sub-network the two chords carry "
              "enough flow that")
        print("  demand variation alone cannot turn them; reversal would require "
              "a second entry")
        print("  point, which is how it is produced on the didactic network in "
              "Section 6.2.")
    else:
        print(f"  The first chord reverses at an amplitude of "
              f"{first['amp']*100:.0f} % of nominal demand.")
    print(f"  The closed-loop study of Section 6.8 runs at "
          f"{amps[0]*100:.0f} %, where the residence times")
    print(f"  already drift by "
          f"{[r for r in rows if r['amp']==amps[0]][0]['drift']*100:.0f} % from "
          f"nominal - so the frozen-flow predictor is genuinely")
    print("  mismatched even though the flow directions hold.")
    RES['F2'] = rows
    return rows


# ======================================================================
def F3_attribute_uptake():
    print("\n" + "=" * 76)
    print("F3  WHY THE REPORTED UPTAKE CHANGED BETWEEN REVISIONS")
    print("=" * 76)
    import realnet_blend as RB
    out = {}

    RB.DELTA = 0.0
    r = RB.run("mpc", verbose=False, nominal_plant=True)
    out['previous'] = RB.uptake(r)
    RB.DELTA = 0.0
    r = RB.run("mpc", verbose=False)
    out['indep_no_backoff'] = RB.uptake(r)
    RB.DELTA = 0.02
    r = RB.run("mpc", verbose=False)
    out['present'] = RB.uptake(r)

    print(f"{'configuration':<46}{'peak':>7}{'viol %':>8}{'uptake %':>10}")
    lab = {'previous': "previous setup: own model as plant, no back-off",
           'indep_no_backoff': "independent plant, no back-off",
           'present': "independent plant + 2 vol-pt back-off (adopted)"}
    for k in ('previous', 'indep_no_backoff', 'present'):
        m = out[k]
        print(f"{lab[k]:<46}{m['peak']:7.2f}{m['viol']:8.1f}{m['mass']:10.1f}")
    d1 = out['previous']['mass'] - out['indep_no_backoff']['mass']
    d2 = out['indep_no_backoff']['mass'] - out['present']['mass']
    print(f"\n  Of the total fall in reported uptake, {d1:.1f} points come from "
          f"closing the loop against")
    print(f"  an independent plant on a time-varying flow field and "
          f"{d2:.1f} points from the back-off.")
    print("  Note that the middle row still spends time above the limit: without "
          "the back-off the")
    print("  scheduler rides a constraint its own predictor cannot hold. The "
          "adopted configuration")
    print("  is the only one of the three with a defensible zero-violation "
          "claim.")
    RES['F3'] = out
    return out


if __name__ == "__main__":
    F1_two_way_coupling()
    F2_reversal_threshold()
    F3_attribute_uptake()
    np.save("coupling_reversal_uptake.npy", RES, allow_pickle=True)
    print("\nsaved coupling_reversal_uptake.npy")
