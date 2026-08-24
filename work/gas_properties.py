"""
gas_properties.py
-----------------
Thermophysical properties of natural-gas / hydrogen (NG-H2) mixtures used in the
reduced-order RLC modelling study. Isothermal transmission-pipeline assumptions.

All SI units: Pa, K, kg, m, s, mol.

References for property values:
  * CH4 / H2 molar masses, gas constant : CODATA / NIST
  * Isothermal wave speed a^2 = Z R_s T  (standard for gas-pipeline transients,
    Zlotnik, Chertkov & Backhaus, CDC 2015)
"""

import numpy as np

# Universal gas constant
R_UNIV = 8.314462618  # J/(mol*K)

# Component molar masses (kg/mol)
M_H2 = 2.01588e-3
M_CH4 = 16.043e-3
# Pipeline-quality natural gas is not pure methane; a representative transmission
# gas molar mass (incl. ethane/CO2/N2) is ~18 g/mol.
M_NG = 18.0e-3

# Dynamic viscosity of components at ~288 K (Pa*s)
MU_H2 = 8.9e-6
MU_NG = 1.10e-5

# Higher heating value, volumetric at STP (MJ/m^3)
HHV_H2_VOL = 12.75
HHV_NG_VOL = 38.5

# Compressibility factor at transmission pressure (50-70 bar, ~288 K), representative
Z_H2 = 1.0
Z_NG = 0.90


def mixture_properties(x_h2, T=288.15, p_ref=6.0e6):
    """
    Compute NG-H2 mixture properties for a given hydrogen VOLUME fraction x_h2.

    Parameters
    ----------
    x_h2 : float   hydrogen fraction by volume (0..1). For ideal-gas mixing at
                   constant T,p the volume fraction equals the mole fraction.
    T    : float   temperature [K]
    p_ref: float   reference pressure [Pa] used for density/viscosity evaluation

    Returns
    -------
    dict with keys: M, Rs, Z, a, a2, rho, mu, hhv_vol, hhv_ratio, wobbe_ratio
    """
    y = x_h2  # mole fraction (= volume fraction for ideal gas)

    # Molar mass of the mixture
    M = y * M_H2 + (1.0 - y) * M_NG
    # Specific gas constant
    Rs = R_UNIV / M
    # Compressibility factor (linear mixing rule, adequate at these pressures)
    Z = y * Z_H2 + (1.0 - y) * Z_NG
    # Isothermal wave speed  a = sqrt(Z Rs T)
    a2 = Z * Rs * T
    a = np.sqrt(a2)
    # Density at reference pressure  rho = p / (Z Rs T)
    rho = p_ref / (Z * Rs * T)
    # Mixture viscosity (mole-fraction weighted, adequate for friction estimate)
    mu = y * MU_H2 + (1.0 - y) * MU_NG

    # Volumetric higher heating value (per m^3 at STP) - mole/volume weighted
    hhv_vol = y * HHV_H2_VOL + (1.0 - y) * HHV_NG_VOL
    hhv_ratio = hhv_vol / HHV_NG_VOL

    # Wobbe index ~ HHV / sqrt(relative density); relative density ~ M/M_air
    M_air = 28.96e-3
    wobbe = hhv_vol / np.sqrt(M / M_air)
    wobbe_ng = HHV_NG_VOL / np.sqrt(M_NG / M_air)
    wobbe_ratio = wobbe / wobbe_ng

    return dict(x_h2=y, M=M, Rs=Rs, Z=Z, a=a, a2=a2, rho=rho, mu=mu,
                hhv_vol=hhv_vol, hhv_ratio=hhv_ratio, wobbe_ratio=wobbe_ratio)


def darcy_friction(rho, mu, D, mdot, eps=3.0e-5):
    """
    Darcy friction factor via Colebrook-White (turbulent), solved by fixed-point.

    rho  : density [kg/m^3]
    mu   : dynamic viscosity [Pa*s]
    D    : diameter [m]
    mdot : mass flow [kg/s]
    eps  : absolute roughness [m] (commercial steel ~ 0.03 mm)
    """
    A = np.pi * D**2 / 4.0
    v = abs(mdot) / (rho * A + 1e-30)
    Re = rho * v * D / mu + 1e-9
    if Re < 2300:
        return 64.0 / Re
    # Colebrook-White fixed point
    f = 0.02
    for _ in range(50):
        rhs = -2.0 * np.log10(eps / (3.7 * D) + 2.51 / (Re * np.sqrt(f)))
        f_new = 1.0 / rhs**2
        if abs(f_new - f) < 1e-10:
            break
        f = f_new
    return f


def swamee_jain(rho, mu, D, mdot, eps=3.0e-5):
    """Non-iterative turbulent Darcy factor (Swamee-Jain) - fast, velocity-dependent."""
    import numpy as _np
    A = _np.pi * D**2 / 4.0
    v = _np.abs(mdot) / (rho * A + 1e-30)
    Re = rho * v * D / mu + 1e-9
    if Re < 2300:
        return 64.0 / Re
    return 0.25 / (_np.log10(eps/(3.7*D) + 5.74/Re**0.9))**2


if __name__ == "__main__":
    print(f"{'H2 vol%':>8} {'M[g/mol]':>9} {'Rs':>7} {'Z':>5} {'a[m/s]':>8} "
          f"{'rho[kg/m3]':>10} {'HHV[MJ/m3]':>10} {'HHV%':>6} {'Wobbe%':>7}")
    for x in [0.0, 0.05, 0.25, 0.50, 1.0]:
        pr = mixture_properties(x)
        print(f"{x*100:8.0f} {pr['M']*1000:9.2f} {pr['Rs']:7.1f} {pr['Z']:5.2f} "
              f"{pr['a']:8.1f} {pr['rho']:10.2f} {pr['hhv_vol']:10.2f} "
              f"{pr['hhv_ratio']*100:6.1f} {pr['wobbe_ratio']*100:7.1f}")
    # friction sanity
    pr = mixture_properties(0.0)
    f = darcy_friction(pr['rho'], pr['mu'], 1.0, 150.0)
    print(f"\nDarcy f (NG, D=1m, 150 kg/s): {f:.4f}")
