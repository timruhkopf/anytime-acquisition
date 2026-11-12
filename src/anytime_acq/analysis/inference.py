"""Decision rule of notebook 10 (pfn_bo_inference; research_log/decisions/0007 §4 test 2): is the frozen PFN's
predictive miscalibrated on BO contexts (explanation E2)?"""

import pandas as pd

COV_TOL = 0.03  # absolute tolerance on interval coverage (90% and 50%)
ECE_TOL = 0.02  # absolute tolerance on the ECE of an improvement event, BO minus random design
ECE_MAX = 0.03  # an event's ECE on BO contexts below this counts as calibrated


def _get(cal: pd.DataFrame, pt: str, m: str, design: str = "bo") -> pd.Series:
    return cal.loc[(cal.design == design) & (cal.ptype == pt) & (cal.metric == m)].iloc[0]


def _con(con: pd.DataFrame, pt: str, m: str) -> pd.Series:
    return con.loc[(con.ptype == pt) & (con.metric == m)].iloc[0]


def miscalibrated(cal: pd.DataFrame, con: pd.DataFrame, ks: list[int], pt: str) -> bool:
    """BO-context predictive at points `pt` is miscalibrated: a coverage off nominal by >= COV_TOL with its CI
    excluding nominal, or an improvement event whose ECE exceeds the random design's by >= ECE_TOL (CI > 0)."""
    for m, nom in (("cov90", 0.90), ("cov50", 0.50)):
        r = _get(cal, pt, m)
        if abs(r.value - nom) >= COV_TOL and not (r.lo <= nom <= r.hi):
            return True
    for k in ks:
        c = _con(con, pt, f"ece_k{k}")
        if c["diff"] >= ECE_TOL and c.lo > 0:
            return True
    return False


def calibrated(cal: pd.DataFrame, con: pd.DataFrame, ks: list[int], pt: str) -> bool:
    """BO-context predictive at points `pt` is calibrated within tolerance: coverage CIs inside nominal +- COV_TOL,
    and for every event the BO ECE CI below ECE_MAX and the BO - random ECE CI below ECE_TOL."""
    for m, nom in (("cov90", 0.90), ("cov50", 0.50)):
        r = _get(cal, pt, m)
        if not (nom - COV_TOL <= r.lo and r.hi <= nom + COV_TOL):
            return False
    return all(_get(cal, pt, f"ece_k{k}").hi < ECE_MAX and _con(con, pt, f"ece_k{k}").hi < ECE_TOL for k in ks)


def inference_verdict(cal: pd.DataFrame, con: pd.DataFrame, ks: list[int]) -> str:
    """GO (E2 supported: the PFN is miscalibrated on BO contexts, at all points pooled or at EI's picks);
    NO-GO (E2 not supported: calibrated within tolerance on both); otherwise INCONCLUSIVE."""
    pts = ("all", "ei_pick")
    if any(miscalibrated(cal, con, ks, pt) for pt in pts):
        return "GO"
    if all(calibrated(cal, con, ks, pt) for pt in pts):
        return "NO-GO"
    return "INCONCLUSIVE"
