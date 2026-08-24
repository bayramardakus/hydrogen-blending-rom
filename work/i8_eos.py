"""I8 - validate the linear-mixing gas properties against the GERG-2008 EOS.

Two corrections relative to the previous version of this script:

 (1) BASE-GAS CONSISTENCY.  `gas_properties.py` represents pipeline-quality
     H-gas by M_NG = 18.0 g/mol (methane plus ethane / N2 / CO2).  The previous
     comparison evaluated GERG for a *pure methane* + H2 mixture
     (M = 16.04 g/mol).  The two gases therefore differ by 12.2 % in molar mass
     before any mixing rule is applied, and the resulting "14.7 % density error"
     was almost entirely that base-gas mismatch - not a deficiency of the linear
     mixing rule.  Here the GERG mixture is specified with the same base gas.

 (2) INDEPENDENT WAVE SPEED.  The previous comparison formed
         a_GERG = sqrt(Z_GERG * Rs_model * T)
     i.e. it borrowed the model's own specific gas constant.  That makes the
     wave-speed check algebraically identical to half the Z error and blind to
     the molar-mass model.  Here the reference wave speed is taken directly
     from the EOS density,  a_GERG = sqrt(p / rho_GERG),  which is the actual
     isothermal wave speed of the mixture.

Both the pure-CH4 and the consistent base gas are reported so the origin of the
earlier discrepancy is transparent.
"""
import numpy as np
from gas_properties import mixture_properties, R_UNIV, M_H2, M_NG, M_CH4
from CoolProp.CoolProp import AbstractState, PT_INPUTS

T = 288.15
p = 60e5   # 60 bar ABSOLUTE

# Representative H-gas composition consistent with M_NG = 18.0 g/mol.
BASE = dict(Methane=0.885, Ethane=0.040, Nitrogen=0.045, CarbonDioxide=0.030)


def gerg(y_h2, base="blend"):
    """GERG-2008 (CoolProp HEOS) density, Z and molar mass of the mixture."""
    if base == "ch4":
        st = AbstractState("HEOS", "Methane&Hydrogen")
        st.set_mole_fractions([1 - y_h2, y_h2])
    else:
        names = "&".join(BASE) + "&Hydrogen"
        x = np.array(list(BASE.values()), dtype=float)
        x = x / x.sum() * (1 - y_h2)
        st = AbstractState("HEOS", names)
        st.set_mole_fractions(list(x) + [y_h2])
    st.update(PT_INPUTS, p, T)
    rho = st.rhomass()
    M = st.molar_mass()
    Z = p * M / (rho * R_UNIV * T)
    return rho, Z, M


def table(base, title):
    print(f"\n--- {title} ---")
    print(f"{'H2%':>4} | {'M_mod':>6} {'M_EOS':>6} | {'Z_mod':>7} {'Z_EOS':>7} "
          f"{'dZ%':>6} | {'rho_mod':>8} {'rho_EOS':>8} {'drho%':>6} | "
          f"{'a_mod':>6} {'a_EOS':>6} {'da%':>6}")
    mx = dict(Z=0.0, rho=0.0, a=0.0)
    rows = []
    for pct in [0, 5, 10, 15, 20, 25]:
        y = pct / 100
        d = mixture_properties(y, T=T, p_ref=p)
        rg, Zg, Mg = gerg(y, base)
        ag = np.sqrt(p / rg)                       # true isothermal wave speed
        dZ = abs(d['Z'] - Zg) / Zg * 100
        dr = abs(d['rho'] - rg) / rg * 100
        da = abs(d['a'] - ag) / ag * 100
        for k, v in (('Z', dZ), ('rho', dr), ('a', da)):
            mx[k] = max(mx[k], v)
        rows.append(dict(pct=pct, Z=d['Z'], Zg=Zg, rho=d['rho'], rg=rg,
                         a=d['a'], ag=ag, dZ=dZ, dr=dr, da=da))
        print(f"{pct:>4} | {d['M']*1e3:6.2f} {Mg*1e3:6.2f} | {d['Z']:7.4f} "
              f"{Zg:7.4f} {dZ:5.2f}% | {d['rho']:8.2f} {rg:8.2f} {dr:5.2f}% | "
              f"{d['a']:6.1f} {ag:6.1f} {da:5.2f}%")
    print(f"     max over 0-25 vol% H2:  Z {mx['Z']:.2f}% , "
          f"density {mx['rho']:.2f}% , isothermal wave speed {mx['a']:.2f}%")
    return mx, rows


if __name__ == "__main__":
    print("=== I8  Linear mixing vs GERG-2008 "
          f"(T={T} K, p={p/1e5:.0f} bar absolute) ===")
    print(f"model base gas : M_NG = {M_NG*1e3:.2f} g/mol "
          f"(H-gas incl. C2H6/N2/CO2)")
    print(f"EOS base gas   : {BASE}  ->  M = {gerg(0.0)[2]*1e3:.2f} g/mol")

    mx_b, rows = table("blend", "CONSISTENT base gas (M = 18.0 g/mol) - "
                                "the comparison reported in the paper")
    mx_c, _ = table("ch4", "Pure CH4 base gas (M = 16.04 g/mol) - shown only to "
                           "document the origin of the earlier 14.7 % figure")

    print("\nCONCLUSION")
    print("  With a consistently specified base gas the linear mixing rules "
          "reproduce GERG-2008 to")
    print(f"    Z            : {mx_b['Z']:.1f} %")
    print(f"    density      : {mx_b['rho']:.1f} %")
    print(f"    wave speed a : {mx_b['a']:.1f} %")
    print(f"  over 0-25 vol% H2. Against a pure-CH4 reference the density error "
          f"is {mx_c['rho']:.1f} %,")
    print("  of which ~12 % is the base-gas molar-mass difference and not a "
          "mixing-rule error.")
    np.save("i8_eos.npy", dict(blend=mx_b, ch4=mx_c, rows=rows),
            allow_pickle=True)
