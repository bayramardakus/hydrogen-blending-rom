"""fig_parity.py - parity plot of the code-to-code verification against
pandapipes on the real SciGRID_gas sub-network (replaces fig_regen.py, whose
blend-control panel is now produced by make_new_figures.py).
"""
import os as _os
import numpy as np
import figstyle as fs
from figstyle import PALETTE as C

fs.apply_style()
FIGDIR = _os.environ.get(
    "PAPER_FIGS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ms", "figs"))
_os.makedirs(FIGDIR, exist_ok=True)


def main():
    r = np.load("realnet_pandapipes.npy", allow_pickle=True).item()
    fig, ax = fs.panels(2)

    Ppp, Pour = r['P_pp'][1:], r['P_our'][1:]
    lo, hi = min(Ppp.min(), Pour.min()) - 0.5, max(Ppp.max(), Pour.max()) + 0.5
    rmse_p = float(np.sqrt(np.mean((Pour - Ppp) ** 2)))
    ax[0].plot([lo, hi], [lo, hi], '--', color=C['mut'], lw=0.9,
               label="Identity ($y=x$)")
    ax[0].fill_between([lo, hi], [lo - rmse_p, hi - rmse_p],
                       [lo + rmse_p, hi + rmse_p], color=C['rlc'], alpha=0.18,
                       lw=0, label=r"$\pm$RMSE (%.3f bar)" % rmse_p)
    ax[0].scatter(Ppp, Pour, s=16, color=C['rlc'], edgecolor='white', lw=0.4,
                  zorder=3, label="Node pressures")
    ax[0].set_xlabel("pandapipes [bar]")
    fs.ylabel(ax[0], "This work [bar]")
    fs.title(ax[0], 'a', f"Node pressures (MAPE {r['mape_p']:.2f}%)")
    fs.legend_below(ax[0], ncol=2, y=-0.30)

    Qpp, Qour = np.abs(r['Q_pp']), np.abs(r['Q_our'])
    hi = max(Qpp.max(), Qour.max()) * 1.05
    rmse_q = float(np.sqrt(np.mean((Qour - Qpp) ** 2)))
    ax[1].plot([0, hi], [0, hi], '--', color=C['mut'], lw=0.9,
               label="Identity ($y=x$)")
    ax[1].fill_between([0, hi], [-rmse_q, hi - rmse_q], [rmse_q, hi + rmse_q],
                       color=C['grn'], alpha=0.18, lw=0,
                       label=r"$\pm$RMSE (%.3f kg/s)" % rmse_q)
    ax[1].scatter(Qpp, Qour, s=16, color=C['grn'], edgecolor='white', lw=0.4,
                  zorder=3, label="Pipe flows")
    ax[1].set_xlabel("pandapipes [kg/s]")
    fs.ylabel(ax[1], "This work [kg/s]")
    fs.title(ax[1], 'b', f"Pipe flows (MAPE {r['mape_q']:.3f}%)")
    fs.legend_below(ax[1], ncol=2, y=-0.30)

    fs.save(fig, _os.path.join(FIGDIR, "fig_pandapipes.png"), legend_rows=2)


if __name__ == "__main__":
    main()
