"""Verification: every number the manuscript quotes is checked against the
result file it came from.

Run from this directory. The manuscript path may be given as the only
argument; it defaults to ../ms/manuscript.tex.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(HERE, '..', 'ms', 'manuscript.tex'))
s = open(TEX, encoding='utf-8').read()

ok = 0
bad = []


def check(name, condition, detail=''):
    global ok
    if condition:
        ok += 1
        print('  ok   %s' % name)
    else:
        bad.append(name)
        print('  FAIL %s  %s' % (name, detail))


def intext(pattern):
    return re.search(pattern, s) is not None


# ----------------------------------------------------------------- caches
sweep = {round(r['h2']): r for r in np.load('sweep.npy', allow_pickle=True)}
real = {round(r['x'] * 100): r for r in np.load('realnet_reduction.npy', allow_pickle=True)}
pp = np.load('realnet_pandapipes.npy', allow_pickle=True).item()
ub = np.load('uncertainty_budget.npy', allow_pickle=True).item()
rob = np.load('robustness_studies.npy', allow_pickle=True).item()
scal = list(np.load('scal.npy', allow_pickle=True))
blend = np.load('realnet_blend_results.npy', allow_pickle=True).item()

print('accuracy metrics')
check('didactic RMSE range 0.028 to 0.040 bar',
      abs(sweep[0]['rmse'] - 0.028) < 0.001 and abs(sweep[25]['rmse'] - 0.040) < 0.001,
      '%.4f %.4f' % (sweep[0]['rmse'], sweep[25]['rmse']))
check('didactic R2 0.9983 to 0.9981',
      abs(sweep[0]['pr2'] - 0.9983) < 5e-5 and abs(sweep[25]['pr2'] - 0.9981) < 5e-5,
      '%.6f %.6f' % (sweep[0]['pr2'], sweep[25]['pr2']))
check('didactic flow R2 0.99998 at every blend',
      all(abs(r['qr2'] - 0.99998) < 5e-6 for r in sweep.values()))
check('real-network RMSE 0.096 to 0.130 bar',
      abs(real[5]['rmse'] - 0.096) < 0.001 and abs(real[25]['rmse'] - 0.130) < 0.001,
      '%.4f %.4f' % (real[5]['rmse'], real[25]['rmse']))
check('real-network R2 0.9987 to 0.9986',
      abs(real[5]['pr2'] - 0.9987) < 5e-5 and abs(real[25]['pr2'] - 0.9986) < 5e-5,
      '%.6f %.6f' % (real[5]['pr2'], real[25]['pr2']))
check('real-network flow RMSE 1.02 to 1.09 kg/s',
      abs(real[5]['frmse'] - 1.02) < 0.01 and abs(real[25]['frmse'] - 1.09) < 0.01,
      '%.3f %.3f' % (real[5]['frmse'], real[25]['frmse']))

P1 = np.asarray(pp['P_our'], float)[1:]
P2 = np.asarray(pp['P_pp'], float)[1:]
e = P1 - P2
prmse = float(np.sqrt(np.mean(e ** 2)))
pr2 = float(1 - np.sum(e ** 2) / np.sum((P2 - P2.mean()) ** 2))
check('parity RMSE 0.15 bar and R2 0.995',
      abs(prmse - 0.15) < 0.01 and abs(pr2 - 0.995) < 0.001,
      '%.4f %.4f' % (prmse, pr2))

print('uncertainty budget')
base = ub['base_peak']
c = ub['cases']
shift = lambda a, b: max(abs(c[a]['peak'] - base), abs(c[b]['peak'] - base)) * 100
rough = shift('roughness low', 'roughness high')
temp = shift('temperature low', 'temperature high')
dens = shift('density low (EOS)', 'density high (EOS)')
check('roughness 0.015 points', abs(rough - 0.015) < 0.002, '%.4f' % rough)
check('temperature 0.187 points', abs(temp - 0.187) < 0.002, '%.4f' % temp)
check('EOS density 0.063 points', abs(dens - 0.063) < 0.002, '%.4f' % dens)
total = rough + temp + dens + 0.57
check('budget sums to 0.84 points', abs(total - 0.84) < 0.01, '%.3f' % total)
check('nominal peak 18.57 vol%', abs(base * 100 - 18.57) < 0.01, '%.3f' % (base * 100))
check('worst-case min pressure 62.9 bar',
      abs(c['roughness high']['pmin_bar'] - 62.86) < 0.05,
      '%.2f' % c['roughness high']['pmin_bar'])
check('nominal min pressure 65.6 bar',
      abs(c['nominal']['pmin_bar'] - 65.61) < 0.05,
      '%.2f' % c['nominal']['pmin_bar'])
check('tau_max 81 h', abs(c['nominal']['tau_max_h'] - 81.18) < 0.1,
      '%.2f' % c['nominal']['tau_max_h'])

print('robustness')
r1 = {int(r['m']): r for r in rob['R1']}
check('sparse telemetry 2.66 against 2.61 kg/s',
      abs(r1[3]['q_mean'] - 2.66) < 0.01 and abs(r1[21]['q_mean'] - 2.61) < 0.01,
      '%.3f %.3f' % (r1[3]['q_mean'], r1[21]['q_mean']))
check('sparse telemetry pressure 0.032 against 0.027 bar',
      abs(r1[3]['p_mean'] - 0.032) < 0.001 and abs(r1[21]['p_mean'] - 0.027) < 0.001,
      '%.4f %.4f' % (r1[3]['p_mean'], r1[21]['p_mean']))
r2 = {round(r['lvl'] * 100): r for r in rob['R2']}
check('forecast error peak 18.57 to 18.44 vol%',
      abs(r2[0]['peak'] - 18.57) < 0.02 and abs(r2[30]['peak'] - 18.44) < 0.02,
      '%.3f %.3f' % (r2[0]['peak'], r2[30]['peak']))
check('forecast error margin at least 1.43 points',
      min(r['margin'] for r in rob['R2']) >= 1.42,
      '%.3f' % min(r['margin'] for r in rob['R2']))
check('no violation at any forecast level',
      all(r['viol'] == 0 for r in rob['R2']))

print('cost')
big = scal[-1]
check('100 nodes, 5849 km, assembly 24 ms, solve 0.25 s',
      big['N'] == 100 and abs(big['km'] - 5849) < 1
      and abs(big['t_build'] - 24) < 0.5 and abs(big['t_qp'] / 1000 - 0.25) < 0.005,
      '%d %.0f %.2f %.3f' % (big['N'], big['km'], big['t_build'], big['t_qp'] / 1000))
check('factor 436 inside the 120 s interval',
      abs(120.0 / ((big['t_build'] + big['t_qp']) / 1000) - 436) < 3,
      '%.0f' % (120.0 / ((big['t_build'] + big['t_qp']) / 1000)))

# grid convergence of the real-network pressure reference
sa = np.load('structural_analysis.npy', allow_pickle=True).item()
s2 = [r for r in sa['S2'] if abs(r['pct'] - 0.5) < 1e-9][0]
gc = np.load('gridconv.npy', allow_pickle=True).item()
check('pressure reference converged at 5 cells',
      abs(gc[10]['mape'] - gc[5]['mape']) / gc[5]['mape'] < 0.02,
      'MAPE %.4f -> %.4f %%' % (gc[5]['mape'], gc[10]['mape']))

# repeatability over independent draws of the two random inputs
ens = np.load('ensemble.npy', allow_pickle=True)
pk = np.array([r['peak'] for r in ens]); vi = np.array([r['viol'] for r in ens])
check('ensemble of 20 realisations, none violating',
      len(ens) == 20 and vi.max() == 0.0,
      '%d runs, max violation %.3f %%' % (len(ens), vi.max()))
check('ensemble peak 18.30 +/- 0.15, worst 18.59',
      abs(pk.mean() - 18.30) < 0.02 and abs(pk.std(ddof=1) - 0.15) < 0.02
      and abs(pk.max() - 18.59) < 0.02,
      'mean %.3f sd %.3f max %.3f' % (pk.mean(), pk.std(ddof=1), pk.max()))

print('manuscript text')
for pat, name in [
        (r'RMSE 0\.096 to\s+0\.130', 'real RMSE quoted'),
        (r'\$R\^\{2\}\$ falling from 0\.9987 to 0\.9986', 'real R2 quoted'),
        (r'RMSE 0\.15\\,bar, \$R\^\{2\}=0\.995\$', 'parity metrics quoted'),
        (r'0\.84', 'budget total quoted'),
        (r'2\.66\\,kg/s against 2\.61\\,kg/s', 'sparse telemetry quoted'),
        (r'18\.57 to\s*\n?18\.44\\,vol', 'forecast result quoted'),
        (r'N_\{\\mathrm\{c\}\}', 'cell-count symbol renamed'),
]:
    check(name, intext(pat))

check('no em dash', '---' not in s)
# The article stands on its own. Text that refers to an earlier draft, and
# text whose job is to defend the author rather than to qualify a result,
# should not appear in it. Both forms are detected here.
VERSION_TALK = r'''(?ix)
    previous\s+version | earlier\s+version | previous\s+draft | earlier\s+draft
  | previous\s+submission | earlier\s+submission | first\s+submission
  | reported\s+previously | as\s+previously | our\s+previous | previously\s+reported
  | the\s+choice\s+made\s+(in|earlier) | in\s+this\s+revision | this\s+revision\s+of
  | \bprevious\w* | \bearlier\b | \bformerly\b | \bhitherto\b
  | quoted\s+previously | results\s+are\s+new
  | the\s+revised\s+manuscript | the\s+referee | the\s+reviewer'''
DEFENSIVE = r'''(?ix)
    rather\s+than\s+asserted | rather\s+than\s+quietly | rather\s+than\s+avoided
  | rather\s+than\s+neglected | rather\s+than\s+guessed | rather\s+than\s+by\s+oversight
  | describe\s+this\s+honestly | we\s+report\s+both | is\s+stated\s+rather\s+than
  | is\s+reported\s+rather\s+than | so\s+the\s+origin\s+of\s+the\s+discrepancy'''
body = s[:s.index('\\begin{thebibliography}')] if '\\begin{thebibliography}' in s else s
check('no previous-version commentary',
      not re.search(VERSION_TALK, body),
      str([m.group(0) for m in re.finditer(VERSION_TALK, body)][:4]))
check('no sentences that defend the author rather than qualify a result',
      not re.search(DEFENSIVE, body),
      str([m.group(0) for m in re.finditer(DEFENSIVE, body)][:4]))
# the observer study of Section 6.9: these numbers were never machine-checked
# before, which is how the manuscript and the shipped result file could have
# drifted apart without anything failing
check('observer, legacy mean flow error',
      '%.0f' % s2['legacy'][1] in body, '%.0f' % s2['legacy'][1])
check('observer, legacy error as a multiple of the largest pipe flow',
      'seventeen times' in body and 16.5 <= s2['legacy'][1] / 207.67 < 17.5,
      '%.2f' % (s2['legacy'][1] / 207.67))
check('observer, physical tuning error and its share of the mean pipe flow',
      '%.1f' % s2['fixed'][1] in body
      and '%.1f' % (s2['fixed'][1] / 72.153 * 100) in body,
      '%.4f kg/s, %.2f %%' % (s2['fixed'][1], s2['fixed'][1] / 72.153 * 100))
check('observer, improvement is three orders of magnitude',
      'three orders of magnitude' in body
      and 1e3 <= s2['legacy'][1] / s2['fixed'][1] < 1e4,
      '%.0f' % (s2['legacy'][1] / s2['fixed'][1]))

labs = set(re.findall(r'\\label\{(eq:[^}]*)\}', s))
refs = set(re.findall(r'\\eqref\{(eq:[^}]*)\}', s))
check('every numbered equation cited', labs == refs,
      'uncited %s' % sorted(labs - refs))
figs = re.findall(r'\\label\{(fig:[^}]*)\}', s)
tabs = re.findall(r'\\label\{(tab:[^}]*)\}', s)
allrefs = set(re.findall(r'\\ref\{([^}]*)\}', s))
check('every figure and table cited',
      all(x in allrefs for x in figs + tabs),
      '%s' % [x for x in figs + tabs if x not in allrefs])
keys = re.findall(r'\\bibitem\{([^}]*)\}', s)
cited = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', s):
    cited.update(k.strip() for k in m.group(1).split(','))
check('every reference cited and every citation resolved',
      set(keys) == cited, 'uncited %s' % [k for k in keys if k not in cited])

print('\n%d checks passed, %d failed' % (ok, len(bad)))
if bad:
    print('failed:', bad)
sys.exit(1 if bad else 0)
