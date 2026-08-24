"""make_new_figures.py - figures revised in Revision 2.

Every figure here is drawn through `figstyle`, so it shares one type size, one
palette, bracketed units, sentence-case labels, and a frameless legend BELOW the
axes. Threshold and limit lines appear as legend entries rather than as
free-floating text inside the axes; that is what removes the label/data overlaps
present in the previous version of these plots.

Produces
  fig_transport_verification.png   discretisation convergence, front smearing,
                                   frozen-flow envelope
  fig_backoff.png                  constraint back-off versus uptake
  fig_estimator.png                observer robustness under transducer noise
  fig_blend_control.png            didactic blend control
  fig_realnet_blend.png            Case Study 3 blend scheduling
  fig_realnet_pressure.png         Case Study 3 pressure regulation
  fig_robustness.png               sparse telemetry; forecast error
"""
import os as _os
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
from figstyle import PALETTE as C

fs.apply_style()

FIGDIR = _os.environ.get(
    "PAPER_FIGS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


def _p(name):
    return _os.path.join(FIGDIR, name)


# ======================================================================
def fig_transport_verification():
    from gas_properties import mixture_properties
    from network import Network, steady_state
    from composition_adv import build_adv_composition
    from transport_fir import discretise

    conv = np.load("advection_convergence.npy", allow_pickle=True).item()
    stru = np.load("structural_analysis.npy", allow_pickle=True).item()

    net = Network()
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)

    # Two rows, not three side-by-side panels. Panel (c) carries ten scenario
    # names on its y-axis; squeezed into one third of a 190 mm canvas it pushed
    # its own title and its legend off the right-hand edge of the figure, and
    # left panels (a) and (b) with axes too narrow for their legends. Giving (c)
    # the full width fixes all three at once.
    fig = plt.figure(figsize=(fs.W3, 6.2))
    # Explicit margins: panel (c)'s scenario names sit OUTSIDE its axes, and
    # figstyle.save() crops at x=0 to pin the canvas to the column width, so any
    # label that runs past the left edge is lost. hspace is generous because
    # each panel of row 1 carries a two-row legend below it.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          left=0.235, right=0.985, top=0.96, bottom=0.15,
                          hspace=0.62, wspace=0.30)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, :])]

    # ---- (a) convergence of the peak error ----------------------------
    rows = [r for r in conv['rows'] if isinstance(r['M'], (int, np.integer))]
    Ms = [r['M'] for r in rows]
    pk = [max(r['peak'].values()) for r in rows]
    cfl = [r for r in conv['rows'] if r['M'] == 'CFL'][0]
    cfl3 = [r for r in conv['rows'] if r['M'] == 'CFL/3'][0]

    ax[0].loglog(Ms, pk, 'o-', color=C['rlc'], ms=3.5, label=r"Uniform $M$")
    ax[0].plot([cfl['ns'] / net.n_pipes], [max(cfl['peak'].values())], '*',
               ms=9, color=C['acc'], ls='none',
               label=r"CFL-matched $M_k$")
    ax[0].plot([cfl3['ns'] / net.n_pipes], [max(cfl3['peak'].values())], 'D',
               ms=3.5, color=C['grn'], ls='none',
               label=r"CFL-matched, $T_s/3$")
    ax[0].axhline(2.0, color=C['mut'], ls=':', label="Back-off, 2 vol-pts")
    # The two values the text quotes, marked on the curve itself. The labels are
    # offset AWAY from the curve (up and to the right) and given an opaque
    # backing box: placed below-right, as before, they landed underneath the
    # next marker and were unreadable.
    for M, dx, dy in ((1, 9, -13), (4, 9, -13)):
        v = [r for r in rows if r['M'] == M][0]
        ax[0].annotate(f"$M$={M}", (M, max(v['peak'].values())),
                       textcoords="offset points", xytext=(dx, dy),
                       fontsize=7, color="0.30", ha="left", va="center",
                       bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    ax[0].set_xlabel(r"Cells per pipe $M$")
    fs.ylabel(ax[0], "Peak error [vol-pts]")
    fs.title(ax[0], 'a', "Numerical diffusion of the peak")
    fs.legend_below(ax[0], ncol=2, y=-0.30, fontsize=7.0)

    # ---- (b) front smearing -------------------------------------------
    pulse = lambda t: 0.30 if 4 * 3600 <= t < 10 * 3600 else 0.05
    te = conv['te']
    for M, col, ls in [(1, C['ol'], '--'), (4, C['acc'], '-.'),
                       (20, C['grn'], ':')]:
        c = build_adv_composition(net, props, P0, Q0, M=M)
        Ad, Bd = discretise(c['A'], c['B'], te[1] - te[0])
        X = np.zeros((c['nstate'], len(te)))
        X[:, 0] = 0.05
        for k in range(len(te) - 1):
            X[:, k + 1] = Ad @ X[:, k] + Bd[:, 0] * pulse(te[k])
        ax[1].plot(te / 3600, (c['Wsel'][3] @ X) * 100, ls, color=col,
                   label=f"$M$ = {M}")
    ax[1].plot(te / 3600, conv['ref_pulse'][3] * 100, '-', color=C['ref'],
               lw=1.6, label="Plug-flow reference")
    ax[1].set_xlabel("Time [h]")
    fs.ylabel(ax[1], "H$_2$ at N3 [vol%]")
    fs.title(ax[1], 'b', "Front smearing, 6 h pulse")
    fs.legend_below(ax[1], ncol=2, y=-0.30, fontsize=7.0)

    # ---- (c) frozen-flow envelope --------------------------------------
    sc = stru['S3']
    y = np.arange(len(sc))
    ax[2].barh(y - 0.20, [s['mean'] for s in sc], 0.38, color=C['mut'],
               label="Mean over time")
    ax[2].barh(y + 0.20, [s['max'] for s in sc], 0.38, color=C['acc'],
               label="True worst case")
    lab = []
    for s in sc:
        t = (s['label'].replace(" (paper scenario)", "")
                       .replace("secondary entry at ", "")
                       .replace("nominal ", "").replace("heavy loop load  ", ""))
        t = t.replace(", -", " entry \u2212").replace("+", "demand +")
        t = t.replace("d2 ", "N2 ").replace("d5 ", "N5 ")
        lab.append(t + (f"  [{s['rev']} rev.]" if s['rev'] else ""))
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(lab, fontsize=7.0)
    ax[2].set_xlabel("Delivered-blend error [vol-pts]")
    ax[2].grid(axis='y', alpha=0)
    fs.title(ax[2], 'c', "Frozen-flow validity envelope, didactic network")
    fs.legend_below(ax[2], ncol=2, y=-0.22, fontsize=7.0)

    fs.save(fig, _p("fig_transport_verification.png"), legend_rows=2,
            tight=False)


# ======================================================================
def fig_backoff():
    back = np.load("backoff_study.npy", allow_pickle=True)
    D = [r['D'] * 100 for r in back]

    fig, ax = fs.panels(2)

    ax[0].plot(D, [r['mass'] for r in back], 'o-', color=C['rlc'], ms=4,
               label="Green-H$_2$ uptake, mass basis")
    ax[0].set_xlabel(r"Constraint back-off $\Delta$ [vol-pts]")
    fs.ylabel(ax[0], "Uptake [%]")
    fs.title(ax[0], 'a', "Cost of the back-off")
    fs.legend_below(ax[0], ncol=1, y=-0.30)

    ax[1].plot(D, [r['peak'] for r in back], 's--', color=C['acc'], ms=4,
               label="Peak delivered blend")
    ax[1].axhline(20.0, color=C['ol'], ls=':',
                  label="Interchangeability limit, 20 vol%")
    viol = [r for r in back if r['viol'] > 0]
    if viol:
        ax[1].plot([r['D'] * 100 for r in viol], [r['peak'] for r in viol],
                   'o', ms=8, mfc='none', mec=C['ol'], mew=1.3, ls='none',
                   label="Limit exceeded")
    ax[1].set_xlabel(r"Constraint back-off $\Delta$ [vol-pts]")
    fs.ylabel(ax[1], "Peak H$_2$ [vol%]")
    fs.title(ax[1], 'b', "Safety margin obtained")
    fs.legend_below(ax[1], ncol=2, y=-0.30, fontsize=7.0)

    fs.save(fig, _p("fig_backoff.png"), legend_rows=2)


# ======================================================================
def fig_estimator():
    from gas_properties import mixture_properties
    from network import steady_state, build_rlc
    from estimator import build_observer
    import realnet

    stru = np.load("structural_analysis.npy", allow_pickle=True).item()
    rows = stru['S2']

    net = realnet.RealNetwork(72.0, 500.0)
    props = mixture_properties(0.05)
    P0, Q0 = steady_state(net, props)
    A, B, Cm, info = build_rlc(net, props, P0, Q0)
    nN, ns, Ts = net.n_nodes, info['ns'], 120.0
    ob = build_observer(A, B, Cm, P0, Q0, net, Ts)
    q_ref = ob['q_ref']

    fig, ax = fs.panels(3)
    pct = [r['pct'] for r in rows]

    ax[0].loglog(pct, [r['legacy'][1] for r in rows], 'o--', color=C['ol'],
                 ms=3.5, label=r"Previous tuning ($R_v=10^{-3}I$, states in Pa)")
    ax[0].loglog(pct, [r['fixed'][1] for r in rows], 's-', color=C['grn'],
                 ms=3.5, label=r"This work ($R_v=\sigma_v^2 I$, scaled model)")
    ax[0].axhline(np.abs(Q0).max(), color=C['ref'], ls=':',
                  label=f"Largest true pipe flow, {np.abs(Q0).max():.0f} kg/s")
    ax[0].set_xlabel(r"Transducer noise [% of reading, $1\sigma$]")
    fs.ylabel(ax[0], "Flow error [kg/s]")
    fs.title(ax[0], 'a', "Unmeasured flow states")
    fs.legend_below(ax[0], ncol=1, y=-0.30, fontsize=7.0)

    ax[1].loglog(pct, [r['legacy'][0] for r in rows], 'o--', color=C['ol'],
                 ms=3.5, label="Previous tuning")
    ax[1].loglog(pct, [r['fixed'][0] for r in rows], 's-', color=C['grn'],
                 ms=3.5, label="This work")
    ax[1].set_xlabel(r"Transducer noise [% of reading, $1\sigma$]")
    fs.ylabel(ax[1], "Pressure error [bar]")
    fs.title(ax[1], 'b', "Measured pressure states")
    fs.legend_below(ax[1], ncol=1, y=-0.30, fontsize=7.0)

    # (c) recovery from a wrong FLOW estimate, under measurement noise
    rng = np.random.default_rng(1)
    nstep = 90
    for lbl, col, ls, cls in [("0.10 % transducer", C['grn'], '-', 0.001),
                              ("0.25 % transducer", C['rlc'], '--', 0.0025),
                              ("0.50 % transducer", C['ol'], '-.', 0.005)]:
        obx = build_observer(A, B, Cm, P0, Q0, net, Ts, sensor_class=cls)
        Adx, Cdx, Lx = obx['Ad'], obx['Cd'], obx['L']
        sx = cls * P0[1:]
        e = np.zeros(nstep)
        for _ in range(20):
            x = np.zeros(ns)
            xh = np.zeros(ns)
            xh[nN - 1:] += 20.0 / q_ref
            for k in range(nstep):
                e[k] += np.abs(x[nN - 1:] - xh[nN - 1:]).max() * q_ref / 20.0
                y = Cdx @ x + rng.normal(0, sx) / obx['p_ref']
                xh = xh + Lx @ (y - Cdx @ xh)
                x = Adx @ x
                xh = Adx @ xh
        ax[2].semilogy(np.arange(nstep) * Ts / 3600, e / 20, ls, color=col,
                       label=lbl)
    ax[2].set_xlabel("Time [h]")
    fs.ylabel(ax[2], "Normalised error [-]")
    fs.title(ax[2], 'c', "Recovery, 20 kg/s offset")
    fs.legend_below(ax[2], ncol=1, y=-0.30, fontsize=7.0)

    fs.save(fig, _p("fig_estimator.png"), legend_rows=3)


# ======================================================================
def _blend_panels(d, deliv_label, xlim_b, fname, west=False):
    mpc, rea = d['mpc'], d['rea']
    t = mpc['t']
    fig, ax = fs.panels(2)

    if west:
        ax[0].fill_between(t, 5.0, mpc['aE'] * 100, color="#DCE7F2", step="mid",
                           label="Available, east")
        ax[0].fill_between(t, 5.0, mpc['aW'] * 100, color="#DDEEDD", step="mid",
                           alpha=0.8, label="Available, west")
        ax[0].step(t, mpc['uE'] * 100, where="mid", color=C['rlc'],
                   label="MPC injection, east")
        ax[0].step(t, mpc['uW'] * 100, where="mid", color=C['grn'], ls="--",
                   label="MPC injection, west")
        ncol_a, rows_a = 2, 2
    else:
        ax[0].fill_between(t, 5.0, mpc['avail'] * 100, color="0.88", step="mid",
                           label="Available green H$_2$")
        ax[0].step(t, mpc['u'] * 100, where="mid", color=C['rlc'],
                   label="MPC injection")
        ax[0].step(t, rea['u'] * 100, where="mid", color=C['ol'], ls="--",
                   label="Full-injection baseline")
        ncol_a, rows_a = 2, 2
    ax[0].set_xlim(0, xlim_b)
    ax[0].set_xlabel("Time [h]")
    fs.ylabel(ax[0], "Injected H$_2$ [vol%]")
    fs.title(ax[0], 'a', "Injection schedule")
    fs.legend_below(ax[0], ncol=ncol_a, y=-0.30, fontsize=7.0)

    ax[1].plot(t, rea['ydel'].max(axis=1) * 100, '--', color=C['ol'],
               label="Full-injection baseline")
    ax[1].plot(t, mpc['ydel'].max(axis=1) * 100, '-', color=C['rlc'],
               label="Composition-aware MPC")
    ax[1].axhline(20, color=C['acc'], ls=':',
                  label="Interchangeability limit, 20 vol%")
    ax[1].axhline(mpc['y_lim'] * 100, color=C['grn'], ls="-.",
                  label="Constraint with back-off")
    ax[1].set_xlim(0, xlim_b)
    ax[1].set_xlabel("Time [h]")
    fs.ylabel(ax[1], "Delivered H$_2$ [vol%]")
    fs.title(ax[1], 'b', f"Delivered blend, {deliv_label}")
    fs.legend_below(ax[1], ncol=2, y=-0.30, fontsize=7.0)

    fs.save(fig, _p(fname), legend_rows=2)


def fig_blend():
    d = np.load("blend_adv_results.npy", allow_pickle=True).item()
    _blend_panels(d, "worst of 5 nodes", 72, "fig_blend_control.png")


def fig_realnet_blend():
    d = np.load("realnet_blend_results.npy", allow_pickle=True).item()
    _blend_panels(d, "worst of 21 nodes", 120,
                  "fig_realnet_blend.png", west=True)


# ======================================================================
def fig_realnet_pressure():
    p = np.load("realnet_pressure_results.npy", allow_pickle=True).item()
    fig, ax = fs.panels(2)

    ax[0].plot(p['ol']['t'], p['ol']['pmin'], '--', color=C['ol'],
               label="Uncontrolled")
    ax[0].plot(p['cl']['t'], p['cl']['pmin'], '-', color=C['rlc'],
               label="Pressure MPC")
    ax[0].axhline(60, color=C['acc'], ls=':', label="Delivery floor (60 bar)")
    ax[0].set_xlabel("Time [h]")
    fs.ylabel(ax[0], "Min. pressure [bar]")
    fs.title(ax[0], 'a', "Pressure regulation under a contingency")
    fs.legend_below(ax[0], ncol=2, y=-0.30, fontsize=7.0)

    ax[1].plot(p['cl']['t'], p['cl']['psrc'], '-', color=C['rlc'],
               label="MPC compressor set-point")
    ax[1].plot(p['ol']['t'], p['ol']['psrc'], '--', color=C['ol'],
               label="Uncontrolled")
    ax[1].axhline(84, color=C['mut'], ls=':',
                  label="Outlet rating [70\u201384 bar]")
    ax[1].axhline(70, color=C['mut'], ls=':')
    ax[1].set_xlabel("Time [h]")
    fs.ylabel(ax[1], "Outlet pressure [bar]")
    fs.title(ax[1], 'b', "Control action")
    fs.legend_below(ax[1], ncol=2, y=-0.30, fontsize=7.0)

    fs.save(fig, _p("fig_realnet_pressure.png"), legend_rows=2)


# ======================================================================
def fig_robustness():
    r = np.load("robustness_studies.npy", allow_pickle=True).item()
    r1, r2 = r['R1'], r['R2']
    fig, ax = fs.panels(2)

    m = [x['m'] for x in r1]
    ax[0].plot(m, [x['q_mean'] for x in r1], 'o-', color=C['rlc'], ms=4,
               label="Mean")
    ax[0].plot(m, [x['q_max'] for x in r1], 's--', color=C['ol'], ms=4,
               label="Worst case")
    ax[0].axhline(0.03 * 72, color=C['mut'], ls=':',
                  label="3% of mean pipe flow")
    ax[0].set_ylim(0, None)
    ax[0].set_xlabel("Instrumented nodes [of 21]")
    fs.ylabel(ax[0], "Flow error [kg/s]")
    fs.title(ax[0], 'a', "Observer under sparse telemetry")
    fs.legend_below(ax[0], ncol=2, y=-0.30, fontsize=7.0)

    lv = [x['lvl'] * 100 for x in r2]
    ax[1].plot(lv, [x['peak'] for x in r2], 'o-', color=C['rlc'], ms=4,
               label="Peak delivered blend")
    ax[1].axhline(20, color=C['acc'], ls=':',
                  label="Interchangeability limit, 20 vol%")
    ax[1].axhline(18, color=C['grn'], ls='-.',
                  label="Constraint with back-off")
    ax[1].set_ylim(14, 21.5)
    ax[1].set_xlabel(r"Green-H$_2$ forecast error [%, $1\sigma$ AR(1)]")
    fs.ylabel(ax[1], "Peak H$_2$ [vol%]")
    fs.title(ax[1], 'b', "Scheduler under forecast error")
    fs.legend_below(ax[1], ncol=2, y=-0.30, fontsize=7.0)

    fs.save(fig, _p("fig_robustness.png"), legend_rows=2)


if __name__ == "__main__":
    fig_transport_verification()
    fig_backoff()
    fig_estimator()
    fig_blend()
    fig_realnet_blend()
    fig_realnet_pressure()
    fig_robustness()
    print(f"\nfigures written to {_os.path.abspath(FIGDIR)}")
