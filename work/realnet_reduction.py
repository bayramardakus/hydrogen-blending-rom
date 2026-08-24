"""realnet_reduction.py — Case Study 3 reduction-accuracy check.
44-state RLC vs high-fidelity 1-D transient reference (M=5 cells/pipe ->
22 + 23*11 = 275 states) on a demand surge, across 5-25 vol% hydrogen.

The reference is stiff (acoustic modes on short pipes); it is integrated with
BDF using the exact Jacobian sparsity pattern for tractable cost.
"""
import time, numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
from gas_properties import mixture_properties
import realnet
from network import steady_state, build_rlc, build_reference

T_END = 7200.0
# Disturbance nodes selected by rule (two largest offtakes), not by index.
_SN = realnet.major_offtakes(realnet.RealNetwork(72.0, 500.0), 2)
SURGE = [(_SN[0], 90.0, 1800.0, 300.0), (_SN[1], 55.0, 4200.0, 300.0)]

def demand_profile(t, net):
    d = net.demand_nom.astype(float).copy()
    for node, size, start, ramp in SURGE:
        d[node] += size*np.clip((t-start)/ramp, 0, 1)
    return d

def _sparsity(rhs, y0, net):
    n=len(y0); f0=rhs(0.0,y0,net.p_source_nom,net.demand_nom)
    S=lil_matrix((n,n))
    for j in range(n):
        yp=y0.copy(); yp[j]+=max(abs(y0[j])*1e-4,1e-2)
        fp=rhs(0.0,yp,net.p_source_nom,net.demand_nom)
        S[np.abs(fp-f0)>0,j]=1
    return S.tocsr()

def run_case(x_h2, net, x0_warm, M_ref=5):
    props = mixture_properties(x_h2)
    P0, Q0 = steady_state(net, props, x0=x0_warm)
    nN, nP = net.n_nodes, net.n_pipes
    d0 = net.demand_nom.copy()
    A_c, B_c, C_out, info = build_rlc(net, props, P0, Q0)
    def rlc_rhs(t, dx):
        du = np.concatenate([[0.0], demand_profile(t,net)-d0]); return A_c@dx + B_c@du
    t_eval = np.linspace(0, T_END, 361)
    def _time(fn, n=1):
        ts=[]
        for _ in range(n): t0=time.perf_counter(); out=fn(); ts.append(time.perf_counter()-t0)
        return np.median(ts), out
    t_rlc, sol_rlc = _time(lambda: solve_ivp(rlc_rhs,[0,T_END],np.zeros(info['ns']),
                            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-6), n=5)
    P_rlc = np.vstack([np.full_like(t_eval,P0[0]), P0[1:,None]+sol_rlc.y[:nN-1,:]])
    Q_rlc = Q0[:,None]+sol_rlc.y[nN-1:,:]
    rhs, init_state, meta = build_reference(net, props, M=M_ref, Q_basis=Q0)
    y0 = init_state(P0,Q0); S=_sparsity(rhs,y0,net)
    def ref_rhs(t,y): return rhs(t,y,net.p_source_nom, demand_profile(t,net))
    t_ref, sol_ref = _time(lambda: solve_ivp(ref_rhs,[0,T_END],y0, t_eval=t_eval,
                            method="BDF", rtol=1e-7, atol=1e-1, jac_sparsity=S))
    # INTEGRITY CHECK. A silently failed reference solve returns fewer time
    # points and an artificially small error/speed-up; one blend in the previous
    # sweep did exactly that. Fail loudly instead.
    if not sol_ref.success or sol_ref.y.shape[1] != len(t_eval):
        raise RuntimeError(
            f"reference solve failed at y={x_h2:.2f}: {sol_ref.message} "
            f"({sol_ref.y.shape[1]}/{len(t_eval)} time points)")
    idxN = meta['idx']['node']
    P_ref = sol_ref.y[idxN,:].copy(); P_ref[net.source,:]=P0[0]
    Q_ref = np.zeros((nP,len(t_eval)))
    for k in range(nP):
        _,face_sl = meta['idx']['pipe'][k]; Q_ref[k,:]=sol_ref.y[face_sl,:].mean(axis=0)
    Pr=P_ref[1:,:]/1e5; Pm=P_rlc[1:,:]/1e5; err=Pm-Pr
    rmse=np.sqrt(np.mean(err**2)); nrmse=rmse/(Pr.max()-Pr.min())*100
    mape=np.mean(np.abs(err)/np.abs(Pr))*100; maxerr=np.max(np.abs(err))
    ferr=Q_rlc-Q_ref
    qsc=np.maximum(np.abs(Q_ref), 0.01*np.abs(Q_ref).max())
    fmape=np.mean(np.abs(ferr)/qsc)*100
    return dict(x=x_h2, a=props['a'], n_ref=meta['n_state'], n_rlc=info['ns'],
                rmse=rmse, nrmse=nrmse, mape=mape, maxerr=maxerr, fmape=fmape,
                t_ref=t_ref, t_rlc=t_rlc, speedup=t_ref/t_rlc, ref_ok=bool(sol_ref.success))

def main():
    net = realnet.RealNetwork(72.0, 500.0)
    P0,Q0 = steady_state(net, mixture_properties(0.05))
    x0w = np.concatenate([P0[1:]/1e5, Q0])
    print("=== Case Study 3 reduction accuracy: 44-state RLC vs 1-D reference ===")
    print(f"{'H2%':>4} {'a':>5} {'nref':>5} {'nrlc':>5} {'pMAPE%':>8} {'pmax_bar':>9} "
          f"{'qMAPE%':>7} {'speedup':>8} {'refWall_s':>9}")
    rows=[]
    for x in [0.05,0.10,0.15,0.20,0.25]:
        r=run_case(x, net, x0w); rows.append(r)
        print(f"{x*100:4.0f} {r['a']:5.0f} {r['n_ref']:5d} {r['n_rlc']:5d} "
              f"{r['mape']:8.4f} {r['maxerr']:9.4f} {r['fmape']:7.3f} {r['speedup']:8.0f} "
              f"{r['t_ref']:9.1f}", flush=True)
    mp=np.mean([r['mape'] for r in rows]); sp=np.median([r['speedup'] for r in rows])
    fq=np.mean([r['fmape'] for r in rows])
    print(f"\nmean pressure MAPE {mp:.4f}%  | mean flow MAPE {fq:.3f}%  | median speedup {sp:.0f}x")
    print(f"reference {rows[0]['n_ref']} states -> reduced {rows[0]['n_rlc']} states")
    np.save("realnet_reduction.npy", rows, allow_pickle=True)

if __name__ == "__main__":
    main()
