"""Confidence intervals and interquartile ranges reported in Sections 6.5 and 6.12
(third revision). Reads the result files already in this package; no simulation
is re-run.

  ensemble.npy         twenty closed-loop realisations (Section 6.12)
  sweep.npy            didactic-network timing sweep (Section 6.5, Table 7)
  realnet_reduction.npy real-network timing sweep (Section 6.5)
"""
import numpy as np
from scipy import stats

e = list(np.load('ensemble.npy', allow_pickle=True))
for key, label in (('peak', 'peak delivered fraction [vol%]'), ('mass', 'uptake by mass [%]')):
    x = np.array([r[key] for r in e], float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    print('%-32s n=%d mean %.2f sd %.2f  95%% CI of mean %.2f to %.2f  range %.2f to %.2f'
          % (label, n, m, s, m - h, m + h, x.min(), x.max()))
pk = np.array([r['peak'] for r in e], float)
print('distance of the ensemble mean below 20 vol%%: %.1f standard deviations' % ((20 - pk.mean()) / pk.std(ddof=1)))
print('one-sided 95%% upper bound on P(violation) from 0 of %d: %.3f' % (len(pk), 1 - 0.05 ** (1 / len(pk))))
for f, label in (('sweep.npy', 'didactic'), ('realnet_reduction.npy', 'real network')):
    sp = np.array([r['speedup'] for r in np.load(f, allow_pickle=True)], float)
    q1, q3 = np.percentile(sp, [25, 75])
    print('speed-up, %-12s median %.0f, interquartile range %.0f to %.0f' % (label, np.median(sp), q1, q3))
