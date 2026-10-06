"""Redraw all twenty figures from the cached result files.

Nothing is re-simulated: every figure is drawn from the .npy files produced by
the original runs, so no reported number can move. Only the type size changes.

Usage: PAPER_FIGS=<dir> python3 regen_figs.py
"""
import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.environ.setdefault('PAPER_FIGS', 'figs')
os.makedirs(FIGDIR, exist_ok=True)

# ---- data plots -------------------------------------------------------------
import make_figures as mf
res5 = np.load('res5.npy', allow_pickle=True).item()
res25 = np.load('res25.npy', allow_pickle=True).item()
mf.fig_properties()
mf.fig_validation(res5, res25)
mf.fig_timing(res5, res25)
mf.fig_network()

import make_new_figures as mnf
mnf.fig_transport_verification()
mnf.fig_backoff()
mnf.fig_estimator()
mnf.fig_blend()
mnf.fig_realnet_blend()
mnf.fig_realnet_pressure()
mnf.fig_robustness()

import fig_parity
fig_parity.main()

import fig_composition            # module-level script
import fig_uncertainty
fig_uncertainty.main()
# fig_uncertainty writes into the working directory, not PAPER_FIGS
import shutil
for ext in ('png', 'pdf'):
    src = 'fig_uncertainty.' + ext
    if os.path.exists(src) and os.path.abspath(FIGDIR) != os.path.abspath('.'):
        shutil.copy(src, os.path.join(FIGDIR, src))

import scalability as sc
sc.make_scal_fig(list(np.load('scal.npy', allow_pickle=True)))

import mpc_analysis as ma
d = np.load('mpc_ana.npy', allow_pickle=True).item()
ma.make_figures(d['Hs'], d['hres'], d['Rs'], d['rres'], d['obs'], d['comp'])

import fig_realnet
fig_realnet.fig_map()
fig_realnet.fig_mpc()

# ---- hand-laid-out schematics ----------------------------------------------
import fig_rlc_circuit           # module-level script
import fig_arch                  # module-level script

print('\nall figures redrawn in', os.path.abspath(FIGDIR))
