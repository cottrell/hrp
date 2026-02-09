# HRP - Fast Hierarchical Risk Parity

Lightweight numpy-only implementation of **Hierarchical Risk Parity** from Marcos Lopez de Prado's "Advances in Financial Machine Learning".

```python
In [1]: import hrp
In [2]: import numpy as np
In [3]: y = np.random.randn(100, 5)
In [4]: result = hrp.get_hrp(y)
In [5]: result.keys()
Out[5]: dict_keys(['y', 'cov', 'corr', 'dist', 'link', 'qd_order', 'qd_corr', 'hrp'])
In [6]: result['hrp']  # portfolio weights
Out[6]: array([0.2, 0.2, 0.2, 0.2, 0.2])
```

## Key Features

- **Fast**: 27-40× faster than popular alternatives (see benchmarks below)
- **Minimal dependencies**: Only numpy and scipy required
- **Simple**: ~140 lines of readable code
- **Tested**: Validated against original Prado implementation

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With dev dependencies (for testing/benchmarking)
uv pip install -e ".[dev]"
```

## Performance Benchmarks

Comparison against popular HRP libraries on random return matrices:

| Implementation | 100×20 | 500×50 | 1000×100 |
|----------------|--------|--------|----------|
| **hrp (numpy)** | **1.03±0.30ms** | **2.44±0.57ms** | **6.61±0.12ms** |
| PyPortfolioOpt | 27.76±1.36ms | 74.25±2.87ms | 183.12±7.16ms |
| Riskfolio-Lib | 39.02±6.31ms | 100.04±3.09ms | 257.30±15.72ms |

*Times shown as mean ± std dev over 100 runs. All implementations produce equivalent results (validated in tests).*

**Speedup summary:**
- 27-30× faster than PyPortfolioOpt
- 38-41× faster than Riskfolio-Lib

### Why so fast?

This implementation:
1. Removes pandas overhead (operates on numpy arrays directly)
2. Uses recursive graph traversal instead of pandas-based quasi-diagonalization
3. Minimizes memory allocations and copies

For comparison with original Prado implementation (pandas-based):
```python
In [132]: %timeit hrp.unlink(link)  # this implementation
12.9 µs ± 328 ns per loop

In [133]: %timeit prado.getQuasiDiag(link)  # original pandas version
7.34 ms ± 341 µs per loop
```

**~570× faster** quasi-diagonalization step.

## Alternatives

While this implementation is optimized for speed, other libraries offer more features:

### [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt)
- **When to use**: Need multiple optimization methods (mean-variance, Black-Litterman, efficient frontier)
- **Pros**: Comprehensive, well-documented, production-ready
- **Cons**: ~30× slower than this implementation
- **Best for**: General portfolio optimization toolkit

### [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/)
- **When to use**: Need advanced risk measures, constraints, visualization
- **Pros**: Extensive features, academic-grade, excellent docs
- **Cons**: ~40× slower, heavier dependencies
- **Best for**: Research and advanced portfolio management

### This Implementation (hrp)
- **When to use**: Speed is critical, need minimal dependencies
- **Pros**: Fastest, simplest, easy to modify
- **Cons**: Only HRP, minimal features
- **Best for**: High-frequency rebalancing, research, learning

## Testing

```bash
# Run basic tests
make test

# Run benchmarks
make benchmark

# Or with pytest directly
pytest hrp -v
pytest hrp/test_benchmark.py -v -s
```

## Algorithm Overview

HRP avoids issues with traditional mean-variance optimization:

1. **Tree Clustering**: Build dendrogram from correlation matrix
2. **Quasi-Diagonalization**: Reorder assets to reveal block structure
3. **Recursive Bisection**: Allocate weights using inverse-variance within clusters

Benefits:
- No covariance matrix inversion (numerically stable)
- Works with singular matrices
- Incorporates asset relationships via clustering

## Development Notes

Quick copy/paste and refactor from Prado's original code:
- Removed pandas dependency for core functionality
- Used recursive functions for graph operations
- Weakly tested, use at own risk (though validated against original)
- PRs welcome, or consider contributing to established alternatives

Original inspiration: Search GitHub for `getRecBiPart` to find similar implementations.

## License

This is a refactored copy/paste from public implementations. Use freely but verify independently.
