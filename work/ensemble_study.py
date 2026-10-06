"""Repeat the Case Study 3 blend loop over independent draws of the two random
inputs the scenario carries, the per-node demand phases and the injector
actuator noise, so that the zero-violation result is reported with a spread
rather than as a single realisation."""
import os
import numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realnet_blend as RB

N = 20
rows = []
for k in range(N):
    RB.DELTA = 0.02
    r = RB.run("mpc", verbose=False, phase_seed=100 + k, act_seed=200 + k)
    m = RB.uptake(r)
    rows.append(dict(k=k, **m))
    print('%3d  peak %7.3f  viol %6.2f%%  mass %6.2f%%  margin %6.3f'
          % (k, m['peak'], m['viol'], m['mass'], m['margin']), flush=True)
    np.save('ensemble.npy', np.array(rows, dtype=object), allow_pickle=True)

pk = np.array([r['peak'] for r in rows]); vi = np.array([r['viol'] for r in rows])
ms = np.array([r['mass'] for r in rows])
print('\npeak   : mean %.3f  sd %.3f  min %.3f  max %.3f' % (pk.mean(), pk.std(ddof=1), pk.min(), pk.max()))
print('viol %% : max %.3f   runs with any violation: %d of %d' % (vi.max(), int((vi > 0).sum()), N))
print('uptake : mean %.2f  sd %.2f  min %.2f  max %.2f' % (ms.mean(), ms.std(ddof=1), ms.min(), ms.max()))
print('worst-case margin to the 20 vol%% limit: %.3f points' % (20.0 - pk.max()))
print('DONE')
