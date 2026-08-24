"""energy_accounting.py — Case Study 3 simple energy balance (R2-C19 / R4-M10).
Compares (i) the chemical energy of green hydrogen absorbed into the grid,
(ii) the parasitic compression work to inject that hydrogen at line pressure, and
(iii) the extra compressor duty of the pressure-regulation MPC — to put the
'control energy' penalty on a physical footing and moderate efficiency claims.
Compression work. Isothermal work w_iso = Rs*T*ln(p2/p1) [J/kg] is the
thermodynamic LOWER bound (the previous version of this file described it, in
error, as a conservative upper bound). Real machines are polytropic with an
efficiency of roughly 0.75-0.85, so the shaft work is w_iso/eta_pol. Both are
reported below; the paper quotes the polytropic figure.
"""
import numpy as np
from gas_properties import mixture_properties
import realnet, realnet_blend as RB

R_UNIV=8.314462618; T=288.15
ETA_POL=0.80          # polytropic efficiency of a transmission compressor
HHV_H2_MASS=141.8e6   # J/kg (higher heating value of hydrogen)
Rs_H2=R_UNIV/2.016e-3
Rs_NG=R_UNIV/mixture_properties(0.0)['M']

def run():
    net,props,P0,Q0=realnet.build()
    bl=np.load('realnet_blend_results.npy',allow_pickle=True).item(); mpc=bl['mpc']
    pr=np.load('realnet_pressure_results.npy',allow_pickle=True).item()
    # (i) green-H2 chemical energy absorbed (two injection points), 24 h
    ts=mpc['t']*3600; MDOT_E,MDOT_W=mpc['MDOT_E'],mpc['MDOT_W']
    def wmass(y): return RB.wmass(y)
    # H2 mass rate above baseline at each injector = (wmass(u)-wmass(base))*throughput
    mdotH2=(np.maximum(RB.wmass(mpc['uE'])-RB.wmass(0.05),0)*MDOT_E
            +np.maximum(RB.wmass(mpc['uW'])-RB.wmass(0.05),0)*MDOT_W)   # kg/s
    E_h2=np.trapezoid(mdotH2*HHV_H2_MASS,ts)          # J over 24 h
    # (ii) injection compression: electrolyser 30 bar -> line 72 bar
    w_inj=Rs_H2*T*np.log(72/30)                       # J/kg H2 (isothermal)
    E_injcomp_iso=np.trapezoid(mdotH2*w_inj,ts)
    E_injcomp=E_injcomp_iso/ETA_POL                   # polytropic shaft work
    # (iii) pressure-MPC extra compressor duty: raise full stream 72 -> psrc(t)
    tsp=pr['cl']['t']*3600; psrc=pr['cl']['psrc']
    w_boost=Rs_NG*T*np.log(np.maximum(psrc/72.0,1.0))/ETA_POL   # J/kg gas
    P_boost=w_boost*500.0                              # W (500 kg/s)
    E_boost=np.trapezoid(P_boost,tsp)
    # gas energy throughput reference (NG HHV ~ 52 MJ/kg mass)
    HHV_NG_MASS=52e6; E_through=500.0*HHV_NG_MASS*(tsp[-1]-tsp[0])
    print("=== CS3 energy accounting ===")
    print(f"(i)  green-H2 chemical energy absorbed (24 h)      = {E_h2/3.6e12:.1f} GWh  "
          f"(mean H2 rate {np.mean(mdotH2):.2f} kg/s)")
    print(f"(ii) injection compression 30->72 bar             = {E_injcomp/3.6e12:.4f} GWh  "
          f"= {E_injcomp/E_h2*100:.2f}% of absorbed H2 energy "
          f"(polytropic, eta={ETA_POL:.2f}; isothermal lower bound "
          f"{E_injcomp_iso/E_h2*100:.2f}%)")
    print(f"(iii)pressure-MPC compressor boost (peak {P_boost.max()/1e6:.1f} MW) = "
          f"{E_boost/3.6e9:.1f} MWh over 2 h = {E_boost/E_through*100:.3f}% of gas energy throughput")
    print(f"     -> injection compression is ~{E_injcomp/E_h2*100:.1f}% parasitic; "
          f"pressure-regulation boost is <{E_boost/E_through*100:.2f}% of throughput energy")
    np.save('energy_accounting.npy',dict(E_h2_GWh=E_h2/3.6e12,inj_parasitic_pct=E_injcomp/E_h2*100,
            boost_peak_MW=P_boost.max()/1e6,boost_pct=E_boost/E_through*100),allow_pickle=True)

if __name__=="__main__":
    run()
