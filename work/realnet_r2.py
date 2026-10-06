"""RMSE and R^2 for the Case Study 3 reduction, computed without disturbing
the reported timings.

`realnet_reduction.py` is re-run for the five blends, but only the accuracy
fields are taken from the new run; `t_ref`, `t_rlc` and `speedup` are carried
over from the cached `realnet_reduction.npy`, so the speed-up quoted in the text
and drawn in the figures continues to come from one single timing run. The
recomputed MAPE, RMSE and maximum error are checked against the cached values
and must agree exactly.
"""
import numpy as np

import realnet
from gas_properties import mixture_properties
from network import steady_state
from realnet_reduction import run_case

CACHE = 'realnet_reduction.npy'


def main():
    cached = {round(r['x'], 3): r
              for r in np.load(CACHE, allow_pickle=True).tolist()}

    net = realnet.RealNetwork(72.0, 500.0)
    P0, Q0 = steady_state(net, mixture_properties(0.05))
    x0w = np.concatenate([P0[1:] / 1e5, Q0])

    print(f"{'H2%':>4} {'p-RMSE[bar]':>12} {'p-MAPE%':>9} {'p-R2':>9} "
          f"{'q-RMSE[kg/s]':>13} {'q-MAPE%':>9} {'q-R2':>9}")
    rows = []
    for x in [0.05, 0.10, 0.15, 0.20, 0.25]:
        r = run_case(x, net, x0w)
        old = cached[round(x, 3)]
        for name, new, ref in (('rmse', r['rmse'], old['rmse']),
                               ('mape', r['mape'], old['mape']),
                               ('maxerr', r['maxerr'], old['maxerr']),
                               ('fmape', r['fmape'], old['fmape'])):
            if abs(float(new) - float(ref)) > 1e-9:
                print(f"  MISMATCH {name} at {x:.2f}: {new!r} vs cached {ref!r}")
        print(f"{x*100:4.0f} {r['rmse']:12.4f} {r['mape']:9.4f} {r['pr2']:9.6f} "
              f"{r['frmse']:13.4f} {r['fmape']:9.4f} {r['qr2']:9.6f}", flush=True)
        row = dict(old)
        row.update(pr2=r['pr2'], qr2=r['qr2'], frmse=r['frmse'])
        rows.append(row)

    np.save(CACHE, rows, allow_pickle=True)
    print('\n%s rewritten with pr2, qr2 and frmse (timings unchanged)' % CACHE)
    print('mean pressure R2 %.6f | mean flow R2 %.6f'
          % (np.mean([r['pr2'] for r in rows]), np.mean([r['qr2'] for r in rows])))


if __name__ == '__main__':
    main()
