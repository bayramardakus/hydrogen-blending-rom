"""Grid convergence of the real-network pressure reference.

Section 6.2 shows the advected composition front needs tens of cells. The
pressure field is diffusive, not advective, so it is expected to resolve at a
far coarser discretisation. This checks that expectation directly: the same
transient is integrated with 5 and with 10 finite-volume cells per pipe, and the
difference between the two references is compared with the reduction error the
paper reports against the 5-cell one.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realnet, realnet_reduction as RR
from network import steady_state

net = realnet.RealNetwork(72.0, 500.0)
out = {}
for M in (5, 10):
    t0 = time.time()
    r = RR.run_case(0.25, net, None, M_ref=M)
    out[M] = r
    print('M=%d  states %s  wall %.0f s  keys %s'
          % (M, r.get('n_ref'), time.time()-t0, sorted(r.keys())), flush=True)
    np.save('gridconv.npy', out, allow_pickle=True)
print('DONE')
