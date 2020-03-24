# FROM CH 16 OF AFML PRADO BOOK

import numpy as np
import scipy.cluster.hierarchy as sch


def get_hrp(y):
    """Hierarchical Risk Parity from Prado.
    Watch out for the ordering! This should return in terms of original index.
    If you want the ordering that prado used, put in a Series like this:
        pd.Series(hrp).loc[qd_order]
    """
    assert isinstance(y, np.ndarray)
    # 1) tree clustering step
    cov = np.cov(y.T)
    corr = np.corrcoef(y.T)
    dist = correl_dist(corr)
    link = sch.linkage(dist, "single")  # see scipy docs for this matrix
    # 2) quasi diagonalization
    qd_order = unlink(link)  # use this to quasi diag (permute) corr
    # sortIx = corr.index[sortIx].tolist()  # recover labels (was a df in prado version)
    qd_corr = corr[qd_order, :][:, qd_order]  # reorder
    # 4) Capital allocation
    hrp = get_rec_bi_part(cov, qd_order)
    return locals()


def get_rec_bi_part(cov, qd_order):
    """Recursive bisection. Numpy only."""
    assert isinstance(cov, np.ndarray)
    assert isinstance(qd_order, list)
    w = np.ones(len(qd_order))
    c_items = [qd_order]
    while len(c_items) > 0:
        c_items_new = list()
        # bi-section
        for i in c_items:
            if len(i) > 1:
                n = int(len(i) / 2)
                for j, k in ((0, n), (n, len(i))):
                    c_items_new.append(i[j:k])
        c_items = c_items_new
        # parse in pairs
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]  # cluster 1
            c_items1 = c_items[i + 1]  # cluster 2
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha  # weight 1
            w[c_items1] *= 1 - alpha  # weight 2
    return w


def get_cluster_var(cov, c_items):
    """Cluster variance"""
    assert isinstance(cov, np.ndarray)
    cov_ = cov[c_items, :][:, c_items]
    w_ = get_ivp(cov_).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var


def get_ivp(cov, **kargs):
    """Inverse-variance portfolio"""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def unlink(link):
    """Quasi diagonalization. Simpler faster way to reverse the graph.
    Check against getQuasiDiag.
        In [132]: %timeit p.unlink(link)                                                                                                                                                                                   
        12.9 µs ± 328 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        In [133]: %timeit p.getQuasiDiag(link)                                                                                                                                                                             
        7.34 ms ± 341 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    link = link.astype(int)
    c = np.arange(link.shape[0]) + link[-1, 3]
    root_id = c[-1]
    d = dict(list(zip(c, link[:, 0:2].tolist())))

    def recursive_unlink(x, d):
        """ Start this with x = root integer """
        if x in d:
            return [xxx for xx in d[x] for xxx in recursive_unlink(xx, d)]
            # return [f(xx, d) for xx in d[x]]
        else:
            return [x]

    return recursive_unlink(root_id, d)


def correl_dist(corr):
    """A distance matrix based on correlation, where 0< = d[i,j]< = 1
    This is a proper distance metric"""
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def test_unlink_against_quasi_diag():
    from .prado_orig import getQuasiDiag
    import random

    y = np.random.randn(100, 20)
    cov = np.cov(y.T)
    corr = np.corrcoef(y.T)
    dist = correl_dist(corr)
    link = sch.linkage(dist, "single")  # see scipy docs for this matrix
    a = getQuasiDiag(link)
    b = unlink(link)
    assert np.allclose(a, b), f"{a} vs {b} did not match"
    print("pass: unlink matches getQuasiDiag")


def test_get_hrp(atol=1e-7):
    import random
    import pandas as pd
    from .prado_orig import getHRP

    y = np.random.randn(100, 20)
    y_df = pd.DataFrame(y)
    a = get_hrp(y)
    b = getHRP(y_df)
    return locals()
    for k in ["cov", "corr", "dist"]:
        assert np.allclose(a[k], b[k].values, atol=atol), f"{k} are different"
    assert np.allclose(a["hrp"], b["hrp"].sort_index().values, f"hrp are different")
    for k in ["link"]:
        assert np.allclose(a[k], b[k], atol=atol), f"{k} are different"
    assert a["qd_order"] == b["sortIx"], f"qd_order != sortIx"
    assert np.allclose(a["qd_corr"], b["df0"].values), f"qd_corr != df0"
    assert a["qd_order"] == b["hrp"].index.tolist(), f"qd_order != hrp.index.tolist()"
    pd.testing.assert_series_equal(pd.Series(a["hrp"]).loc[a["qd_order"]], b["hrp"])
    return locals()
