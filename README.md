# HRP

    In [1]: import hrp
    In [2]: y = np.random.randn(100, 5)
    In [3]: l = hrp.get_hrp(y)
    In [4]: l.keys()
    Out[4]: dict_keys(['y', 'cov', 'corr', 'dist', 'link', 'qd_order', 'qd_corr', 'hrp'])

Quick job copy paste and refactor of Hierarchical Risk Parity (See Prado "Advances in Financial Machine Learning).

    * remove pandas dependency.
    * use recursive functions for graph ops
    * weakly tested, use at own risk.
    * PR's welcome but there are other similar copy/pasta projects too. Search for `getRecBiPart` for example in github code search. Happy to push this into any one of those too.

For example of difference between Prado pandas and numpy version, check unlink vs getQuasiDiag is 1000x:

        In [132]: %timeit p.unlink(link)
        12.9 µs ± 328 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        In [133]: %timeit p.getQuasiDiag(link)
        7.34 ms ± 341 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Test

Test to show things matches output or original code.

    make test
