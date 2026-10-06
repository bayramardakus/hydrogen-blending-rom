"""Accuracy metrics: RMSE and the coefficient of determination alongside
the MAPE.

The blend sweep of `verify.py` is re-run for the six hydrogen fractions, but
only the error metrics are recomputed: the timings and speed-ups already stored
in `sweep.npy` are carried over unchanged, so Table 4 and Fig. 9 continue to be
drawn from one and the same timing run. The recomputed RMSE and MAPE are printed
next to the cached values as a self-check; they must agree to the last digit.

Output: sweep.npy is rewritten with two extra fields per row, `pr2` (pressure)
and `qr2` (flow), and `rmse`/`frmse` in physical units.
"""
import numpy as np

from experiment_validation import run_case

BLENDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def r2(pred, ref):
    """Coefficient of determination, pooled over nodes and time."""
    ss_res = np.sum((pred - ref) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    cached = {round(r['h2'], 3): r
              for r in np.load('sweep.npy', allow_pickle=True).tolist()}
    rows = []
    print(f"{'H2%':>5} {'p-RMSE[bar]':>12} {'p-MAPE%':>9} {'p-R2':>9} "
          f"{'q-RMSE[kg/s]':>13} {'q-MAPE%':>9} {'q-R2':>9}")
    for x in BLENDS:
        r = run_case(x, M_ref=20)
        key = round(x * 100, 3)
        old = cached[key]

        Pr = r['P_ref'][1:, :] / 1e5          # bar, non-source nodes
        Pm = r['P_rlc'][1:, :] / 1e5
        pr2 = r2(Pm, Pr)
        qr2 = r2(r['Q_rlc'], r['Q_ref'])

        # self-check against the cached run
        for name, new, ref in (('rmse', r['rmse'], old['rmse']),
                               ('mape', r['mape'], old['pmape']),
                               ('fmape', r['fmape'], old['qmape'])):
            if abs(new - ref) > 1e-9:
                print(f"  MISMATCH {name} at {key}%: {new!r} vs cached {ref!r}")

        print(f"{key:5.0f} {r['rmse']:12.4f} {r['mape']:9.4f} {pr2:9.6f} "
              f"{r['frmse']:13.4f} {r['fmape']:9.4f} {qr2:9.6f}")

        row = dict(old)
        row.update(pr2=pr2, qr2=qr2, frmse=r['frmse'])
        rows.append(row)

    np.save('sweep.npy', rows, allow_pickle=True)
    print('\nsweep.npy rewritten with pr2, qr2 and frmse '
          '(timings and speed-ups carried over unchanged)')


if __name__ == '__main__':
    main()
