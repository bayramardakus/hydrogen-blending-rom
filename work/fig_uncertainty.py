"""fig_uncertainty.py - the error-analysis figure.

(a) Accuracy of the reduced model against the 1-D reference, with the RMSE
    plotted as an error band about the reduced trace rather than quoted only in
    a table.
(b) Delivered hydrogen fraction at the critical node, with the band swept out by
    the input-parameter ensemble of `uncertainty_budget.py` (pipe roughness, gas
    temperature, equation-of-state density).
(c) What each source contributes to the peak delivered fraction, against the
    constraint back-off that has to cover all of them.

Inputs : res25.npy, uncertainty_budget.npy, sweep.npy
Output : fig_uncertainty.png / .pdf
"""
import matplotlib.pyplot as plt
import numpy as np

import figstyle as FS

TS_H = 0.5          # blend-loop sampling interval [h]
Y_LIMIT = 20.0      # interchangeability cap [vol%]
DELTA = 2.0         # constraint back-off [volume points]
RESIDUAL = 0.57     # measured closed-loop predictor error [volume points]


def main():
    FS.apply_style()
    P = FS.PALETTE

    res = np.load('res25.npy', allow_pickle=True).item()
    ub = np.load('uncertainty_budget.npy', allow_pickle=True).item()
    sweep = {round(r['h2']): r for r in np.load('sweep.npy', allow_pickle=True)}

    # Two rows rather than three panels abreast, so the traces of (a) and
    # (b) and the bars of (c) are all readable at print size.
    fig = plt.figure(figsize=(FS.W3, 6.6))
    # Two grids, not one: panel (c) names its sources on the y-axis, outside
    # its own axes, so it needs a wider left margin than (a) and (b) do.
    gs_top = fig.add_gridspec(1, 2, left=0.095, right=0.985,
                              top=0.96, bottom=0.66, wspace=0.26)
    gs_bot = fig.add_gridspec(1, 1, left=0.345, right=0.985,
                              top=0.40, bottom=0.175)
    ax = [fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1]),
          fig.add_subplot(gs_bot[0, 0])]

    # ---------------------------------------------------------------- (a)
    t = res['t'] / 3600.0
    Pr = res['P_ref'][1:, :] / 1e5
    Pm = res['P_rlc'][1:, :] / 1e5
    n = int(np.argmax(np.abs(Pm - Pr).max(axis=1)))     # worst node
    rmse = sweep[25]['rmse']
    ax[0].plot(t, Pr[n], FS.DASH[0], color=P['ref'], label='1-D reference')
    ax[0].plot(t, Pm[n], FS.DASH[1], color=P['rlc'], label='Reduced model')
    ax[0].fill_between(t, Pm[n] - rmse, Pm[n] + rmse, color=P['rlc'],
                       alpha=0.22, lw=0,
                       label=r'$\pm$RMSE (%.3f bar)' % rmse)
    ax[0].set_xlabel('Time [h]')
    FS.ylabel(ax[0], 'Pressure [bar]')
    FS.title(ax[0], 'a', 'Reduction error band')
    FS.legend_row(ax[0], 3, max_per_row=1)

    # ---------------------------------------------------------------- (b)
    cases = ub['cases']
    nom = cases['nominal']['ydel'] * 100.0
    tb = np.arange(nom.shape[0]) * TS_H
    crit = int(np.argmax(nom.max(axis=0)))
    stack = np.stack([c['ydel'][:, crit] for c in cases.values()]) * 100.0
    lo, hi = stack.min(axis=0), stack.max(axis=0)

    # The active window is the injection day and the transit that follows it;
    # beyond it every trace sits at the base blend. Error bars, not a shaded
    # band, because the spread is small next to the axis range and a band that
    # narrow is invisible in print.
    win = tb <= 72.0
    half = (hi - lo) / 2.0
    ax[1].plot(tb[win], nom[win, crit], FS.DASH[0], color=P['rlc'],
               label='Delivered blend, nominal')
    ax[1].errorbar(tb[win][::6], nom[win, crit][::6], yerr=half[win][::6],
                   fmt='o', ms=2.0, color=P['rlc'], elinewidth=0.8,
                   capsize=1.8, lw=0,
                   label='Input-parameter spread')
    ax[1].axhline(Y_LIMIT, ls=FS.DASH[2], color=P['acc'],
                  label='Interchangeability limit')
    ax[1].axhline(Y_LIMIT - DELTA, ls=FS.DASH[3], color=P['grn'],
                  label='Backed-off limit')
    ax[1].set_xlabel('Time [h]')
    FS.ylabel(ax[1], 'Delivered H$_2$ [vol%]')
    FS.title(ax[1], 'b', 'Delivered blend with spread')
    FS.legend_row(ax[1], 4, max_per_row=1)

    # ---------------------------------------------------------------- (c)
    base = ub['base_peak']
    def shift(a, b):
        return max(abs(cases[a]['peak'] - base), abs(cases[b]['peak'] - base)) * 100

    contrib = [
        ('Roughness, $2$ to $20\\times10^{-5}$ m',
         shift('roughness low', 'roughness high')),
        ('Gas temperature, 263 to 303 K',
         shift('temperature low', 'temperature high')),
        ('Equation of state, $\\pm2.3\\%$ density',
         shift('density low (EOS)', 'density high (EOS)')),
        ('Predictor residual, closed loop', RESIDUAL),
    ]
    names = [c[0] for c in contrib]
    vals = [c[1] for c in contrib]
    ypos = np.arange(len(vals))
    ax[2].barh(ypos, vals, color=P['rlc'], height=0.6,
               label='Contribution')
    ax[2].axvline(DELTA, ls=FS.DASH[2], color=P['acc'],
                  label='Adopted back-off')
    ax[2].axvline(sum(vals), ls=FS.DASH[3], color=P['grn'],
                  label='Sum of contributions')
    ax[2].set_yticks(ypos)
    ax[2].set_yticklabels(names)
    ax[2].invert_yaxis()
    ax[2].set_xlabel('Shift in peak delivered fraction [vol-points]')
    FS.title(ax[2], 'c', 'Uncertainty budget')
    # Two rows: at this type size three entries in one row ran
    # past the right-hand edge of the canvas.
    FS.legend_row(ax[2], 3, max_per_row=2)

    FS.save(fig, 'fig_uncertainty.png', legend_rows=5, tight=False)
    print('worst didactic node index %d, RMSE %.4f bar' % (n, rmse))
    print('critical delivery node index %d, nominal peak %.3f vol%%'
          % (crit, nom[:, crit].max()))
    print('band half-width at the peak: %.3f vol-points'
          % ((hi - lo)[int(np.argmax(nom[:, crit]))] / 2))
    for nme, v in zip([c[0].replace('\n', ' ') for c in contrib], vals):
        print('  %-40s %.3f' % (nme, v))
    print('  %-40s %.3f' % ('SUM', sum(vals)))
    print('  %-40s %.3f' % ('adopted back-off', DELTA))


if __name__ == '__main__':
    main()
