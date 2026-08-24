"""parameter_studies.py - the three parameter sweeps reported in Section 6 that
are not produced by any single case-study script.

They were run ad hoc for the previous revision, which left three result files
(`backoff_study.npy`, `wobbe_variant.npy`, `surge_sweep.npy`) in the repository
with no script that regenerates them. That is exactly the gap the data-and-code
statement promises not to have, so the three sweeps are collected here.

  S1  BACK-OFF SWEEP (Section 4.3, Fig. 15).  Green-hydrogen uptake and peak
      delivered blend against the constraint back-off Delta. Run in the
      INDEPENDENT plant - the same configuration as the headline Case Study 3
      result - so that the Delta = 2 vol-pt row reproduces the uptake reported
      in Section 6.9 exactly. The previous revision's cache was generated
      against an intermediate configuration and did not.

  S2  INTERCHANGEABILITY-CRITERION VARIANT (Section 6.10).  The same scheduler
      against three delivered-blend caps, each the hydrogen fraction at which a
      given Wobbe-index band is reached. Same independent-plant basis as S1, so
      the three criteria are compared on a common footing, and against the same
      plant as every other closed-loop number in the paper.

  S3  CONTINGENCY-MAGNITUDE SWEEP (Section 6.9).  Minimum node pressure with and
      without the pressure MPC, over contingency withdrawals from 30 to
      100 kg/s, establishing where the 84 bar outlet rating saturates.

Run time is dominated by S1 and S2 (seven closed-loop MPC runs).

    python parameter_studies.py            # all three
    python parameter_studies.py s3         # just the surge sweep
"""
import sys
import numpy as np

import realnet_blend as RB


# ======================================================================
def wobbe_cap(band_pct):
    """Hydrogen fraction at which the delivered Wobbe index has fallen by
    `band_pct` per cent relative to pure natural gas."""
    from gas_properties import mixture_properties
    xs = np.linspace(0.0, 0.40, 4001)
    dev = np.array([(mixture_properties(x)['wobbe_ratio'] - 1.0) * 100
                    for x in xs])
    # wobbe_ratio is monotone decreasing in x over this range
    return float(np.interp(-band_pct, dev[::-1], xs[::-1]))


# ======================================================================
def S1_backoff_sweep(deltas=(0.0, 0.01, 0.02, 0.03)):
    print("\n" + "=" * 76)
    print("S1  CONSTRAINT BACK-OFF SWEEP  (independent plant)")
    print("=" * 76)
    print(f"{'Delta [vol-pt]':>15}{'uptake %':>10}{'vol %':>8}"
          f"{'peak vol%':>11}{'viol %':>8}{'margin':>8}")
    rows = []
    d0 = RB.DELTA
    try:
        for D in deltas:
            RB.DELTA = D
            r = RB.run("mpc", verbose=False)
            m = RB.uptake(r)
            m['D'] = D
            rows.append(m)
            print(f"{D*100:15.1f}{m['mass']:10.2f}{m['vol']:8.2f}"
                  f"{m['peak']:11.3f}{m['viol']:8.3f}{m['margin']:8.3f}")
    finally:
        RB.DELTA = d0
    np.save("backoff_study.npy", np.array(rows, dtype=object),
            allow_pickle=True)
    print("\nsaved backoff_study.npy")
    return rows


# ======================================================================
def S2_wobbe_variant(bands=(5.0, 4.5, 3.0)):
    print("\n" + "=" * 76)
    print("S2  INTERCHANGEABILITY CRITERION AS A WOBBE BAND  (independent plant)")
    print("=" * 76)
    caps = [round(wobbe_cap(b), 3) for b in bands]
    print(f"{'Wobbe band':>12}{'equiv. cap vol%':>18}{'uptake %':>10}"
          f"{'peak vol%':>11}{'viol %':>8}")
    rows = []
    y0, d0 = RB.Y_LIMIT, RB.DELTA
    try:
        RB.DELTA = 0.02
        for band, cap in zip(bands, caps):
            RB.Y_LIMIT = cap
            r = RB.run("mpc", verbose=False)
            m = RB.uptake(r)
            m['cap'] = cap
            m['band_pct'] = band
            rows.append(m)
            print(f"{-band:11.1f}%{cap*100:18.1f}{m['mass']:10.2f}"
                  f"{m['peak']:11.3f}{m['viol']:8.3f}")
    finally:
        RB.Y_LIMIT, RB.DELTA = y0, d0
    np.save("wobbe_variant.npy", np.array(rows, dtype=object),
            allow_pickle=True)
    print("\nsaved wobbe_variant.npy")
    return rows


# ======================================================================
def S3_surge_sweep(surges=(30.0, 45.0, 60.0, 80.0, 100.0)):
    """Columns: surge [kg/s], min p uncontrolled [bar], % horizon below floor
    uncontrolled, min p closed-loop [bar], % below floor closed-loop, max
    compressor set-point [bar], median solve time [ms]."""
    import realnet_pressure as RP
    floor = RP.P_MIN_BAR
    print("\n" + "=" * 76)
    print("S3  CONTINGENCY-MAGNITUDE SWEEP")
    print("=" * 76)
    print(f"{'surge kg/s':>11}{'p_min OL':>10}{'viol OL %':>11}"
          f"{'p_min CL':>10}{'viol CL %':>11}{'max p_src':>11}"
          f"{'solve ms':>10}")
    out = []
    for q in surges:
        _, _, ol, cl = RP.run(surge_total=q)
        row = [q, ol['pmin'].min(), (ol['pmin'] < floor - 1e-6).mean() * 100,
               cl['pmin'].min(), (cl['pmin'] < floor - 1e-6).mean() * 100,
               cl['psrc'].max(), float(cl['solve_ms'])]
        out.append(row)
        print(f"{row[0]:11.0f}{row[1]:10.2f}{row[2]:11.2f}"
              f"{row[3]:10.2f}{row[4]:11.2f}{row[5]:11.2f}{row[6]:10.1f}")
    RP.SURGE = RP._mk_surge(RP.SURGE_TOTAL)
    arr = np.array(out)
    np.save("surge_sweep.npy", arr)
    print("\nsaved surge_sweep.npy")
    return arr


# ======================================================================
if __name__ == "__main__":
    which = set(a.lower() for a in sys.argv[1:]) or {"s1", "s2", "s3"}
    if "s1" in which:
        S1_backoff_sweep()
    if "s2" in which:
        S2_wobbe_variant()
    if "s3" in which:
        S3_surge_sweep()
