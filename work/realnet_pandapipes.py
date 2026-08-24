"""realnet_pandapipes.py — Case Study 3 independent validation vs pandapipes
(Fraunhofer IEE) on the real SciGRID_gas German H-gas sub-network."""
import numpy as np
import pandapipes as pp
import network as NW
from gas_properties import R_UNIV
from pandapipes_validation import build_in_pandapipes, pandapipes_effective_density
import realnet

def run():
    net = realnet.RealNetwork(72.0, 500.0)
    ppn, j = build_in_pandapipes(net, p_src_bar=72.0, fluid="hgas")
    pp.pipeflow(ppn)
    P_pp = ppn.res_junction["p_bar"].values.copy()
    Q_pp = ppn.res_pipe["mdot_from_kg_per_s"].values.copy()
    rho_pp = pandapipes_effective_density(ppn, net.diam)
    f = ppn.fluid; T=288.15; mu=f.get_property("viscosity",T)
    M=float(np.atleast_1d(f.get_molar_mass())[0])*1e-3; Rs=R_UNIV/M
    Z=np.mean(P_pp)*1e5/(rho_pp*Rs*T)
    props=dict(a2=Z*Rs*T,a=np.sqrt(Z*Rs*T),rho=rho_pp,mu=mu,Z=Z,Rs=Rs,M=M)
    P0,Q0=NW.steady_state(net,props)
    P_our=P0/1e5
    err_p=P_our[1:]-P_pp[1:]; mape_p=np.mean(np.abs(err_p)/np.abs(P_pp[1:]))*100
    max_p=np.max(np.abs(err_p))
    err_q=np.abs(Q0)-np.abs(Q_pp); qscale=np.maximum(np.abs(Q_pp),0.01*np.abs(Q_pp).max())
    mape_q=np.mean(np.abs(err_q)/qscale)*100
    max_q=np.max(np.abs(err_q))
    print("=== CS3 independent validation vs pandapipes (hgas) ===")
    print(f"  network: {net.n_nodes} nodes, {net.n_pipes} pipes, {net.total_km:.0f} km")
    print(f"  pandapipes eff. density {rho_pp:.2f} kg/m3 (matched Z={Z:.3f}, M={M*1000:.1f} g/mol)")
    print(f"  node-pressure range pandapipes [{P_pp[1:].min():.2f},{P_pp[1:].max():.2f}] bar")
    print(f"  --> pressure MAPE = {mape_p:.4f}%   max = {max_p:.4f} bar")
    print(f"  --> flow     MAPE = {mape_q:.4f}%   max = {max_q:.3f} kg/s")
    np.save("realnet_pandapipes.npy", dict(P_pp=P_pp,P_our=P_our,Q_pp=Q_pp,Q_our=Q0,
            mape_p=mape_p,max_p=max_p,mape_q=mape_q,max_q=max_q,rho=rho_pp,Z=Z,M=M),
            allow_pickle=True)
    return mape_p,mape_q

if __name__=="__main__":
    run()
