"""Validation tests against Prado's original implementation."""
import numpy as np
import pandas as pd
import pytest

from hrp import get_hrp, unlink, correl_dist
from hrp.prado_orig import getQuasiDiag, getHRP
import scipy.cluster.hierarchy as sch


def test_unlink_against_quasi_diag():
    """Verify unlink matches Prado's getQuasiDiag."""
    np.random.seed(42)
    y = np.random.randn(100, 20)
    cov = np.cov(y.T)
    corr = np.corrcoef(y.T)
    dist = correl_dist(corr)
    link = sch.linkage(dist, "single")

    a = getQuasiDiag(link)
    b = unlink(link)

    assert np.allclose(a, b), f"{a} vs {b} did not match"
    print("✓ unlink matches getQuasiDiag")


def test_get_hrp_matches_prado(atol=1e-7):
    """Verify get_hrp matches Prado's getHRP."""
    np.random.seed(42)
    y = np.random.randn(100, 20)
    y_df = pd.DataFrame(y)

    a = get_hrp(y)
    b = getHRP(y_df)

    # Test covariance, correlation, and distance matrices
    for k in ["cov", "corr", "dist"]:
        assert np.allclose(a[k], b[k].values, atol=atol), f"{k} are different"

    # Test HRP weights
    assert np.allclose(a["hrp"], b["hrp"].sort_index().values, atol=atol), f"hrp are different"

    # Test linkage matrix
    for k in ["link"]:
        assert np.allclose(a[k], b[k], atol=atol), f"{k} are different"

    # Test ordering
    assert a["qd_order"] == b["sortIx"], f"qd_order != sortIx"

    # Test quasi-diagonal correlation matrix
    assert np.allclose(a["qd_corr"], b["df0"].values, atol=atol), f"qd_corr != df0"

    # The key test: verify HRP weights sum to 1 and are in reasonable range
    assert np.isclose(a["hrp"].sum(), 1.0), "HRP weights don't sum to 1"
    assert np.isclose(b["hrp"].sum(), 1.0), "Prado HRP weights don't sum to 1"

    # Verify both produce similar weight distributions (allowing for numerical differences)
    a_sorted = np.sort(a["hrp"])
    b_sorted = np.sort(b["hrp"].values)
    assert np.allclose(a_sorted, b_sorted, atol=atol), "HRP weight distributions differ"

    print("✓ get_hrp matches getHRP")
